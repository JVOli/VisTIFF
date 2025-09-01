import io
import base64
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage as ndi
from matplotlib import cm, colors
from pyproj import CRS, Transformer
from shapely.geometry import LineString
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
import plotly.graph_objects as go


# --------------- Helpers: Raster I/O and Rendering ---------------

st.set_page_config(
    page_icon="images/Logo_semtexto.png",
    layout="wide",
)

with st.sidebar:
    st.image("images/Logo.png", width=150)

def open_dataset_from_bytes(file_bytes: bytes) -> rasterio.io.DatasetReader:
    """Open a rasterio dataset from raw bytes using MemoryFile."""
    memfile = MemoryFile(file_bytes)
    return memfile.open()


def get_raster_preview_to_4326(
    ds: rasterio.io.DatasetReader,
    band_indexes: Tuple[int, ...],
    max_size: int = 1024,
    colormap_name: str = "viridis",
    src_crs: Optional[CRS] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Reproject the selected band(s) to EPSG:4326 for visualization and return:
    - image array as uint8 RGBA (H, W, 4) or RGB (H, W, 3)
    - bounds in lat/lon as (min_lon, min_lat, max_lon, max_lat)
    """
    dst_crs = "EPSG:4326"

    # Determine output size cap
    scale = min(max_size / ds.width, max_size / ds.height, 1.0)
    dst_width = max(1, int(ds.width * scale))
    dst_height = max(1, int(ds.height * scale))

    source_crs = src_crs or ds.crs
    transform, _, _ = calculate_default_transform(
        source_crs, dst_crs, ds.width, ds.height, *ds.bounds, dst_width=dst_width, dst_height=dst_height
    )

    if len(band_indexes) == 1:
        band_index = band_indexes[0]
        dst = np.empty((dst_height, dst_width), dtype="float32")
        reproject(
            source=rasterio.band(ds, band_index),
            destination=dst,
            src_transform=ds.transform,
            src_crs=source_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        nodata = ds.nodatavals[band_index - 1] if ds.nodatavals else None
        nodata_mask = np.zeros_like(dst, dtype=bool)
        if nodata is not None:
            nodata_mask |= np.isclose(dst, nodata)

        # Compute robust min/max for visualization
        valid = ~np.isnan(dst) & ~nodata_mask
        if not np.any(valid):
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(dst[valid], [2, 98])
            if not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(dst[valid])), float(np.nanmax(dst[valid]))
                if vmin == vmax:
                    vmax = vmin + 1.0

        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = cm.get_cmap(colormap_name)
        rgba = cmap(norm(dst), bytes=True)  # (H, W, 4) uint8

        # Set alpha 0 for nodata
        if nodata is not None:
            rgba[nodata_mask, 3] = 0
        # Set alpha 0 for NaN as well
        nan_mask = ~np.isfinite(dst)
        if np.any(nan_mask):
            rgba[nan_mask, 3] = 0

        img = rgba
    else:
        # RGB composite: assume indexes are (R, G, B)
        rgb = np.empty((3, dst_height, dst_width), dtype="float32")
        for i, bidx in enumerate(band_indexes):
            reproject(
                source=rasterio.band(ds, bidx),
                destination=rgb[i],
                src_transform=ds.transform,
                src_crs=source_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        # Normalize per-channel to 0..255 for display
        img = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
        for i in range(3):
            band = rgb[i]
            valid = np.isfinite(band)
            if np.any(valid):
                p2, p98 = np.percentile(band[valid], [2, 98])
                if p2 == p98:
                    p98 = p2 + 1.0
                band_scaled = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                band_scaled = np.zeros_like(band)
            img[:, :, i] = (band_scaled * 255).astype(np.uint8)

    # Compute bounds in EPSG:4326 from the output transform
    min_lon = transform.c
    max_lat = transform.f
    max_lon = transform.c + transform.a * dst_width
    min_lat = transform.f + transform.e * dst_height

    return img, (min_lon, min_lat, max_lon, max_lat)


def image_array_to_data_url(img: np.ndarray) -> str:
    """Encode a numpy image array (RGB or RGBA) as a PNG data URL string."""
    mode = "RGBA" if img.shape[2] == 4 else "RGB"
    pil_img = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# --------------- Helpers: Geometry and Sampling ---------------

def densify_line(line: LineString, step: float) -> List[Tuple[float, float]]:
    """
    Return points along a line every `step` units (in line CRS units), including endpoints.
    """
    if step <= 0:
        step = line.length
    num = max(2, int(np.ceil(line.length / step)) + 1)
    distances = np.linspace(0, line.length, num)
    points = [line.interpolate(d) for d in distances]
    return [(p.x, p.y) for p in points]


def compute_profile(
    file_bytes: bytes,
    band_index: int,
    coords_latlon: List[Tuple[float, float]],
    override_crs: Optional[CRS] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample raster values along a polyline defined in lat/lon.
    Returns distances (meters) and values arrays.
    """
    with open_dataset_from_bytes(file_bytes) as ds:
        # Prepare transformers
        crs_dst = override_crs or ds.crs
        if crs_dst is None:
            raise ValueError("CRS do raster ausente. Defina o CRS antes de calcular o perfil.")
        to_raster = Transformer.from_crs("EPSG:4326", crs_dst, always_xy=True)
        to_m = Transformer.from_crs(crs_dst, "EPSG:3857", always_xy=True)

        # Convert to raster CRS
        xs, ys = zip(*[to_raster.transform(lon, lat) for lat, lon in [(lat, lon) for lat, lon in coords_latlon]])
        line_raster = LineString(list(zip(xs, ys)))

        # Determine step ~ 1 pixel
        pixel_size = min(abs(ds.transform.a), abs(ds.transform.e))
        step = pixel_size

        sample_points = densify_line(line_raster, step)

        # Distances in meters along the line
        mx, my = zip(*[to_m.transform(x, y) for x, y in sample_points])
        mx = np.array(mx)
        my = np.array(my)
        dists = np.hstack(([0.0], np.sqrt(np.diff(mx) ** 2 + np.diff(my) ** 2)))
        dists = np.cumsum(dists)

        # Sample raster using dataset's built-in sampler
        values = list(ds.sample(sample_points, indexes=band_index))
        values = np.array([v[0] if v is not None else np.nan for v in values], dtype=float)

    return dists, values


# --------------- Streamlit App ---------------

st.set_page_config(page_title="Perfil Longitudinal em GeoTIFF", layout="wide")
st.title("Visualização de GeoTIFF e Perfis Longitudinais")
st.caption("Carregue arquivos .tiff/.tif, visualize no mapa e desenhe linhas para extrair perfis.")

with st.sidebar:
    st.header("Entrada de dados")
    uploaded_files = st.file_uploader(
        "Envie um ou mais arquivos .tiff/.tif", type=["tif", "tiff"], accept_multiple_files=True
    )
    colormap_name = st.selectbox(
        "Mapa de cores (para banda única)",
        [
            "viridis",
            "terrain",
            "plasma",
            "magma",
            "cividis",
            "gray",
        ],
        index=0,
    )
    smoothing_method = st.selectbox(
        "Suavização do raster (aplicada aos dados)",
        ["Nenhuma", "Gaussiana", "Média (caixa)", "Mediana"],
        index=0,
        help="Aplica filtro diretamente nos dados do raster. Afeta visualização e perfil/vazões."
    )
    gaussian_radius = 0.0
    box_radius = 0
    median_size = 3
    if smoothing_method == "Gaussiana":
        gaussian_radius = st.slider("Raio da Gaussiana", 0.0, 50.0, 1.0, 0.1)
    elif smoothing_method == "Média (caixa)":
        box_radius = st.slider("Raio da média (caixa)", 0, 50, 1, 1)
    elif smoothing_method == "Mediana":
        median_size = st.slider("Tamanho da janela (ímpar)", 3, 50, 3, 2)
    overlay_opacity = st.slider("Opacidade do raster", 0.0, 1.0, 0.5, 0.01)

    st.write("Desenvolvido por: João Vitor Cunha")
    st.write("Contato: joao.vitor@pierpartners.com.br")

if not uploaded_files:
    st.info("Envie ao menos um arquivo GeoTIFF para começar.")
    st.stop()

# Persist raw bytes in session state for reliable re-use
if "rasters" not in st.session_state:
    st.session_state["rasters"] = {}

for f in uploaded_files:
    if f.name not in st.session_state["rasters"]:
        st.session_state["rasters"][f.name] = f.getvalue()

all_names = list(st.session_state["rasters"].keys())
active_name = st.selectbox("Selecione o raster ativo", all_names, index=0)
file_bytes_original = st.session_state["rasters"][active_name]

# Cache para rasters suavizados
if "rasters_smoothed" not in st.session_state:
    st.session_state["rasters_smoothed"] = {}

def _apply_smoothing_to_array(
    arr: np.ndarray,
    method: str,
    gaussian_radius: float,
    box_radius: int,
    median_size: int,
) -> np.ndarray:
    # Normalizar nulos em NaN
    arr = arr.astype(np.float32, copy=False)
    finite_mask = np.isfinite(arr)
    if method == "Gaussiana":
        # Convolução normalizada para ignorar NaN
        data = np.where(finite_mask, arr, 0.0)
        weight = finite_mask.astype(np.float32)
        sigma = max(gaussian_radius, 0.0)
        if sigma == 0.0:
            return arr
        data_blur = ndi.gaussian_filter(data, sigma=sigma, mode="nearest")
        weight_blur = ndi.gaussian_filter(weight, sigma=sigma, mode="nearest")
        out = np.divide(
            data_blur,
            np.maximum(weight_blur, 1e-6),
            out=np.zeros_like(arr, dtype=np.float32),
        )
        out[weight_blur < 1e-6] = np.nan
        return out
    elif method == "Média (caixa)":
        size = int(max(0, box_radius) * 2 + 1)
        if size <= 1:
            return arr
        data = np.where(finite_mask, arr, 0.0)
        weight = finite_mask.astype(np.float32)
        data_blur = ndi.uniform_filter(data, size=size, mode="nearest")
        weight_blur = ndi.uniform_filter(weight, size=size, mode="nearest")
        out = np.divide(
            data_blur,
            np.maximum(weight_blur, 1e-6),
            out=np.zeros_like(arr, dtype=np.float32),
        )
        out[weight_blur < 1e-6] = np.nan
        return out
    elif method == "Mediana":
        size = int(median_size)
        if size < 3:
            return arr
        if size % 2 == 0:
            size += 1
        # Preencher NaN com valores existentes para evitar enviesamento severo
        # Aqui optamos por copiar os dados; NaNs permanecerão NaN nas áreas sem vizinhos válidos após filtro
        filled = np.where(finite_mask, arr, np.nan)
        # scipy.median_filter não ignora NaN; substituímos NaN por valor local via nearest antes
        # Aproximação: usar interpolação nearest por convolução do mask
        # Fallback simples: substituir NaN por média local (box pequena) antes do mediano
        pre_size = 3
        data = np.where(finite_mask, arr, 0.0)
        weight = finite_mask.astype(np.float32)
        local_mean = ndi.uniform_filter(data, size=pre_size, mode="nearest") / np.maximum(
            ndi.uniform_filter(weight, size=pre_size, mode="nearest"), 1e-6
        )
        seed = np.where(finite_mask, arr, local_mean)
        out = ndi.median_filter(seed, size=size, mode="nearest").astype(np.float32)
        # Reaplicar NaN onde não havia dados em uma janela grande
        big_weight = ndi.uniform_filter(weight, size=size, mode="nearest")
        out[big_weight < 1e-6] = np.nan
        return out
    else:
        return arr

def smooth_raster_bytes(
    file_bytes: bytes,
    method: str,
    gaussian_radius: float,
    box_radius: int,
    median_size: int,
) -> bytes:
    with open_dataset_from_bytes(file_bytes) as src:
        meta = src.meta.copy()
        width, height, count = src.width, src.height, src.count
        # Forçar saída em float32 para manter decimais e NaN
        meta.update(dtype="float32", nodata=None)
        bands_out = []
        # Tentamos usar nodata do raster para mapear em NaN
        for b in range(1, count + 1):
            arr = src.read(b).astype(np.float32)
            nodata_val = None
            if src.nodatavals and len(src.nodatavals) >= b:
                nodata_val = src.nodatavals[b - 1]
            if nodata_val is not None:
                arr = np.where(np.isclose(arr, float(nodata_val)), np.nan, arr)
            out = _apply_smoothing_to_array(arr, method, gaussian_radius, box_radius, median_size)
            bands_out.append(out.astype(np.float32))

        mem = MemoryFile()
        with mem.open(
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype="float32",
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            for b in range(count):
                dst.write(bands_out[b], b + 1)
        smoothed_bytes = mem.read()
        mem.close()
        return smoothed_bytes

# Preparar bytes ativos (original ou suavizado)
file_bytes = file_bytes_original
if smoothing_method != "Nenhuma":
    smooth_key = (active_name, smoothing_method, float(gaussian_radius), int(box_radius), int(median_size))
    if smooth_key not in st.session_state["rasters_smoothed"]:
        with st.spinner("Aplicando suavização ao raster..."):
            try:
                st.session_state["rasters_smoothed"][smooth_key] = smooth_raster_bytes(
                    file_bytes_original,
                    smoothing_method,
                    gaussian_radius,
                    box_radius,
                    median_size,
                )
            except Exception as e:
                st.warning(f"Falha ao suavizar raster: {e}")
                st.session_state["rasters_smoothed"][smooth_key] = file_bytes_original
    file_bytes = st.session_state["rasters_smoothed"][smooth_key]

with open_dataset_from_bytes(file_bytes) as ds_meta:
    st.subheader("Metadados do raster ativo")
    cols = st.columns(4)
    cols[0].metric("Largura", ds_meta.width)
    cols[1].metric("Altura", ds_meta.height)
    cols[2].metric("Bandas", ds_meta.count)
    # Permitir override do CRS quando ausente ou inválido
    ds_crs_text = str(ds_meta.crs) if ds_meta.crs else "(não definido)"
    cols[3].metric("CRS", ds_crs_text)

    user_crs_input = st.text_input(
        "CRS (opcional, e.g., EPSG:31983, EPSG:4674)",
        value="" if ds_meta.crs else "EPSG:4326",
        help="Informe um código EPSG válido caso o raster não tenha CRS definido ou esteja incorreto."
    )
    override_crs: Optional[CRS] = None
    if user_crs_input.strip():
        try:
            override_crs = CRS.from_user_input(user_crs_input.strip())
        except Exception:
            st.warning("CRS informado inválido. Utilizando CRS do raster (se existir).")
            override_crs = None

    # Band selection
    band_options = list(range(1, ds_meta.count + 1))
    default_band = 1

    view_mode = "Banda única"
    rgb_possible = ds_meta.count >= 3
    if rgb_possible:
        view_mode = st.radio("Modo de visualização", ["Banda única", "RGB (1,2,3)"], horizontal=True)

    if view_mode == "Banda única":
        band_index = st.selectbox("Selecione a banda", band_options, index=default_band - 1)
        band_indexes = (band_index,)
    else:
        band_indexes = (1, 2, 3)

    # Prepare preview img and bounds
    try:
        img, (min_lon, min_lat, max_lon, max_lat) = get_raster_preview_to_4326(
            ds_meta, band_indexes, max_size=1024, colormap_name=colormap_name, src_crs=override_crs or ds_meta.crs
        )
    except Exception as e:
        st.error(
            "Não foi possível reprojetar o raster para visualização. Informe um CRS válido (ex.: EPSG:4326, EPSG:31983) e tente novamente.\n" 
            f"Detalhe: {e}"
        )
        st.stop()

data_url = image_array_to_data_url(img)

# Build folium map
center_lat = (min_lat + max_lat) / 2.0
center_lon = (min_lon + max_lon) / 2.0

m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
folium.raster_layers.ImageOverlay(
    image=data_url,
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    opacity=overlay_opacity,
    interactive=False,
    cross_origin=False,
    zindex=1,
).add_to(m)

Draw(
    export=False,
    filename="drawn.geojson",
    draw_options={
        "polyline": True,
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "marker": False,
        "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
).add_to(m)

st.subheader("Mapa e seleção de linha")
map_out = st_folium(m, width=None, height=600, returned_objects=["last_active_drawing", "all_drawings"]) 


def extract_last_line(drawing: Optional[Dict]) -> Optional[List[Tuple[float, float]]]:
    if not drawing:
        return None
    geom = drawing.get("geometry") or drawing
    if not geom:
        return None
    gtype = geom.get("type")
    if gtype == "LineString":
        coords = geom.get("coordinates", [])
        # GeoJSON order: [lon, lat]
        latlon = [(c[1], c[0]) for c in coords]
        return latlon
    return None


last_line_latlon: Optional[List[Tuple[float, float]]] = None
if map_out and isinstance(map_out, dict):
    last_line_latlon = extract_last_line(map_out.get("last_active_drawing"))
    if last_line_latlon is None:
        # Try from all_drawings if available
        drawings = map_out.get("all_drawings") or []
        if drawings:
            last_line_latlon = extract_last_line(drawings[-1])

st.markdown("Desenhe uma linha no mapa para extrair o perfil.")

if last_line_latlon:
    st.success("Linha detectada. Calculando perfil...")
    band_index_for_profile = band_indexes[0]
    try:
        distances_m, values = compute_profile(file_bytes, band_index_for_profile, last_line_latlon, override_crs=override_crs)
    except Exception as e:
        st.error(f"Erro ao calcular perfil: {e}")
        distances_m, values = None, None

    if distances_m is not None and values is not None:
        # Estatísticas básicas
        has_valid = np.any(np.isfinite(values))
        if has_valid:
            stats = {
                "Mín": float(np.nanmin(values)),
                "Máx": float(np.nanmax(values)),
                "Média": float(np.nanmean(values)),
            }
            st.write("Estatísticas do perfil:", stats)

        # Nível de referência para área sob o nível
        default_level = float(np.nanmedian(values)) if has_valid else 0.0
        level = st.number_input(
            "Nível/Altura para cálculo de área",
            value=default_level,
            step=0.1,
            help="Valor de referência horizontal. A área calculada considera (nível - perfil) quando o perfil está abaixo do nível.",
        )
        col_n, col_s = st.columns(2)
        with col_n:
            manning_n = st.number_input(
                "Coeficiente de Manning n",
                value=0.030,
                min_value=0.000,
                step=0.005,
                format="%.5f",
                help="Rugosidade. Ex.: 0,012 (canal liso) a 0,05+ (natural rugoso)."
            )
        with col_s:
            slope_s = st.number_input(
                "Declividade hidráulica S",
                value=0.001,
                min_value=0.000,
                step=0.0001,
                format="%.5f",
                help="Declividade de energia (adimensional). Aproximar pela declividade do escoamento."
            )

        # Inserir pontos de interseção com o nível para interpolar corretamente
        x = distances_m.astype(float)
        y = values.astype(float)
        is_finite = np.isfinite(x) & np.isfinite(y)
        x = x[is_finite]
        y = y[is_finite]

        xx: List[float] = []
        yy: List[float] = []
        if len(x) >= 1:
            xx.append(float(x[0]))
            yy.append(float(y[0]))
            for i in range(len(x) - 1):
                xi, yi = float(x[i]), float(y[i])
                xj, yj = float(x[i + 1]), float(y[i + 1])
                # Check crossing with the horizontal level
                di = yi - level
                dj = yj - level
                if di == 0.0:
                    # Point lies exactly on the level, keep as is
                    pass
                if dj == 0.0:
                    # we will add xj,yj later
                    pass
                # If signs differ (one below and one above), insert intersection
                if di * dj < 0:
                    t = di / (di - dj)  # fraction from i to j where y == level
                    xc = xi + t * (xj - xi)
                    # Add crossing at y==level before appending next point
                    xx.append(xc)
                    yy.append(level)
                # Append next original point
                xx.append(xj)
                yy.append(yj)

        if len(xx) == 0:
            xx = x
            yy = y

        y_threshold = np.full_like(np.asarray(xx, dtype=float), level, dtype=float)
        y_area = np.minimum(np.asarray(yy, dtype=float), level)

        fig = go.Figure()
        # Linha horizontal do nível
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=y_threshold,
                mode="lines",
                name="Nível",
                line=dict(color="red", dash="dash"),
            )
        )
        # Área preenchida entre nível e perfil (apenas onde perfil < nível)
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=y_area,
                mode="lines",
                name="Área abaixo do nível",
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.20)",
                line=dict(color="rgba(255,0,0,0)"),
                hoverinfo="skip",
            )
        )
        # Perfil por cima
        fig.add_trace(
            go.Scatter(x=xx, y=yy, mode="lines", name="Perfil", line=dict(color="#1f77b4"))
        )

        fig.update_layout(
            xaxis_title="Distância (m)",
            yaxis_title=f"Valor da banda {band_index_for_profile}",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cálculo da área sob o nível (trapezoidal) com pontos de interseção
        diff = level - y_area
        diff = np.where(np.isfinite(diff), diff, 0.0)
        area_under = float(np.trapz(diff, xx))
        st.metric("Área abaixo do nível", f"{area_under:,.2f} m²")

        # Perímetro molhado: soma dos comprimentos ao longo do leito submerso
        wetted_perimeter = 0.0
        for i in range(len(xx) - 1):
            yi = yy[i]
            yj = yy[i + 1]
            if (yi <= level) and (yj <= level):
                dx = xx[i + 1] - xx[i]
                dy = yj - yi
                wetted_perimeter += float(np.hypot(dx, dy))

        # Raio hidráulico e vazão pela equação de Manning: Q = (1/n) * A * R^(2/3) * S^(1/2)
        discharge_q = None
        hydraulic_radius = None
        if area_under > 0 and wetted_perimeter > 0 and manning_n > 0 and slope_s >= 0:
            hydraulic_radius = area_under / wetted_perimeter
            discharge_q = (1.0 / manning_n) * area_under * (hydraulic_radius ** (2.0 / 3.0)) * (slope_s ** 0.5)

        col_a, col_p, col_r, col_q = st.columns(4)
        col_a.metric("Área molhada A", f"{area_under:,.2f} m²")
        col_p.metric("Perímetro molhado P", f"{wetted_perimeter:,.2f} m")
        col_r.metric("Raio hidráulico R", f"{(hydraulic_radius or 0):,.4f} m")
        col_q.metric("Vazão Q (Manning)", f"{(discharge_q or 0):,.3f} m³/s")
else:
    st.info("Nenhuma linha desenhada ainda.")

