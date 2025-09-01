## VisTIFF — Visualização de GeoTIFF e Perfis Longitudinais

Aplicativo Streamlit para visualizar GeoTIFFs, aplicar suavização diretamente nos dados e extrair perfis longitudinais a partir de linhas desenhadas no mapa, com estatísticas básicas e estimativas hidráulicas (Manning).

### Recursos principais
- **Upload de GeoTIFF**: suporte a múltiplos arquivos `.tif/.tiff`.
- **Visualização em mapa**: reprojeção automática para EPSG:4326 com diferentes mapas de cores.
- **Suavização aplicada aos dados**: Gaussiana, Média (caixa) e Mediana. A suavização afeta tanto a visualização quanto o cálculo do perfil e das vazões.
- **Perfis longitudinais**: desenhe uma linha no mapa e extraia o perfil da banda selecionada.
- **Métricas e hidráulica**: estatísticas básicas, área abaixo de um nível, perímetro molhado, raio hidráulico e vazão via equação de Manning.
- **CRS**: permite informar um CRS manualmente quando ausente/incorreto no raster.

### Requisitos
- Python 3.9+ recomendado
- Sistema operacional testado: Windows 10/11

Dependências principais (vide `requirements.txt`):
- streamlit, streamlit-folium, folium
- rasterio, numpy, matplotlib, Pillow, shapely, pyproj, plotly
- scipy (para filtros de suavização)

### Instalação
No PowerShell (Windows):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Execução

```powershell
streamlit run app.py
```

O app abrirá no navegador padrão (geralmente em `http://localhost:8501`).

### Como usar
1. Envie um ou mais arquivos `.tif/.tiff` na barra lateral.
2. Escolha o raster ativo e, se necessário, informe o CRS (ex.: `EPSG:31983`).
3. Selecione o modo de visualização: **Banda única** (com mapa de cores) ou **RGB (1,2,3)**.
4. Ajuste a **Suavização do raster (aplicada aos dados)** conforme desejado. Os dados do raster serão suavizados em memória e usados em todas as etapas seguintes.
5. No mapa, **desenhe uma linha** (ferramenta Polyline). O perfil será calculado automaticamente.
6. Ajuste o **nível** para cálculo de área, informe **n de Manning** e **declividade S** para obter estimativas hidráulicas.

### Suavização do raster
- Métodos: **Gaussiana**, **Média (caixa)** e **Mediana**.
- Parâmetros:
  - Gaussiana: raio (σ)
  - Média (caixa): raio (tamanho da janela = 2·raio+1)
  - Mediana: tamanho da janela (ímpar)
- A suavização é aplicada diretamente às bandas do raster em `float32`. Áreas sem dados são mantidas como NaN e ficam transparentes na visualização.

### Perfil e hidráulica
- O perfil é muestreado ao longo da linha desenhada, com passo ~1 pixel no CRS do raster.
- Estatísticas básicas: mínimo, máximo e média.
- Área abaixo do nível: integração trapezoidal onde o perfil está abaixo do nível.
- Vazão (Manning): `Q = (1/n) · A · R^(2/3) · S^(1/2)`.

### Dicas de desempenho
- Prefira **raios/janelas menores** em rasters muito grandes para reduzir tempo de processamento.
- Alterar o método ou parâmetros de suavização recarrega/recacheia o raster suavizado em memória.

### Solução de problemas
- "Falha ao suavizar raster": verifique memória disponível e reduza o tamanho da janela/raio.
- Problemas com CRS: preencha o campo de CRS (ex.: `EPSG:4326`, `EPSG:31983`).
- Instalação do rasterio/pyproj no Windows: mantenha `pip`, `setuptools` e `wheel` atualizados antes de `pip install -r requirements.txt`.

### Estrutura do projeto
```
VisTIFF/
  app.py               # Aplicativo Streamlit
  images/              # Logos
  requirements.txt     # Dependências
  README.md            # Este arquivo
```

### Licença
Defina aqui a licença do projeto, se aplicável.

### Contato
Autor: João Vitor Cunha — joao.vitor@pierpartners.com.br


