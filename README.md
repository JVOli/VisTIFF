# Prototipo

## App Streamlit: GeoTIFF + Perfil Longitudinal

Este protótipo permite:
- Enviar arquivos `.tif/.tiff` (um ou vários)
- Visualizar o raster em um mapa web
- Desenhar uma linha e obter o perfil longitudinal dos valores do raster

### Como rodar

1. (Opcional) Crie e ative um ambiente virtual
2. Instale dependências:
```bash
pip install -r requirements.txt
```
3. Rode o app:
```bash
streamlit run app.py
```

No app:
- Faça upload de um ou mais GeoTIFFs
- Escolha o arquivo ativo e a banda (ou RGB 1-2-3)
- Desenhe uma linha no mapa para ver o perfil longitudinal (em metros)

Requisitos: GeoTIFF georreferenciado com CRS conhecido.
