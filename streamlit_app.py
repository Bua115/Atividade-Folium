"""Execute com `streamlit run streamlit_app.py`."""

from html import escape
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from shapely.geometry import Polygon
from streamlit_folium import st_folium


DIRETORIO_PROJETO = Path(__file__).resolve().parent
CAMINHO_RESULTADOS = DIRETORIO_PROJETO / 'votacao_secao_2026_Divinopolis.csv'
CAMINHO_LOCAIS = DIRETORIO_PROJETO / 'locais_secoes_divinopolis_com_bairros.csv'
# Limite municipal oficial do IBGE para Divinópolis (código 3122306).
CAMINHO_LIMITE_MUNICIPAL = DIRETORIO_PROJETO / 'limite_municipal_divinopolis.geojson'
CRS_METRICO = 'EPSG:31983'


@st.cache_data(show_spinner='Carregando resultados e locais...')
def carregar_dados(
    caminho_resultados: str,
    caminho_locais: str,
    assinatura_resultados: tuple[int, int],
    assinatura_locais: tuple[int, int],
):
    df = pd.read_csv(caminho_resultados, sep=';', encoding='utf-8-sig')
    colunas_interesse = [
        'ANO_ELEICAO', 'NR_TURNO', 'NR_ZONA', 'NR_SECAO', 'NM_MUNICIPIO',
        'NM_VOTAVEL', 'DS_CARGO', 'QT_VOTOS', 'NM_LOCAL_VOTACAO',
        'DS_LOCAL_VOTACAO_ENDERECO', 'NR_LOCAL_VOTACAO',
    ]
    df_divinopolis = df.loc[
        df['NM_MUNICIPIO'].astype('string').str.strip().str.upper().eq('DIVINÓPOLIS'),
        colunas_interesse,
    ].copy()
    if df_divinopolis.empty:
        raise ValueError('Nenhum resultado eleitoral de Divinópolis foi encontrado no CSV.')
    arquivo_locais_mapa = Path(caminho_locais)
    if not arquivo_locais_mapa.exists():
        raise FileNotFoundError(f'Arquivo de locais e bairros não encontrado: {arquivo_locais_mapa.resolve()}')

    colunas_resultados = {
        'ANO_ELEICAO', 'NR_TURNO', 'NR_ZONA', 'NR_SECAO', 'NR_LOCAL_VOTACAO',
        'NM_LOCAL_VOTACAO', 'DS_LOCAL_VOTACAO_ENDERECO',
        'DS_CARGO', 'NM_VOTAVEL', 'QT_VOTOS',
    }
    colunas_ausentes = colunas_resultados.difference(df_divinopolis.columns)
    if colunas_ausentes:
        raise ValueError(f'Colunas ausentes nos resultados eleitorais: {sorted(colunas_ausentes)}')

    resultados_presidente = df_divinopolis[
        df_divinopolis['DS_CARGO'].astype('string').str.strip().str.upper().eq('PRESIDENTE')
    ].copy()
    resultados_presidente['ANO_ELEICAO'] = pd.to_numeric(resultados_presidente['ANO_ELEICAO'], errors='coerce')
    anos_disponiveis = resultados_presidente['ANO_ELEICAO'].dropna()
    if anos_disponiveis.empty:
        raise ValueError('Não há resultados presidenciais com ano de eleição válido.')
    ano_eleicao = int(anos_disponiveis.max())

    resultados_presidente['NR_TURNO'] = pd.to_numeric(resultados_presidente['NR_TURNO'], errors='coerce')
    resultados_presidente['QT_VOTOS'] = pd.to_numeric(resultados_presidente['QT_VOTOS'], errors='coerce').fillna(0)
    resultados_presidente = resultados_presidente[
        (resultados_presidente['ANO_ELEICAO'] == ano_eleicao)
        & resultados_presidente['NR_TURNO'].isin([1, 2])
        & resultados_presidente['NM_VOTAVEL'].notna()
    ].copy()
    nomes_votaveis = resultados_presidente['NM_VOTAVEL'].astype('string').str.strip().str.upper()
    votos_brancos_nulos = resultados_presidente[
        nomes_votaveis.isin(['VOTO BRANCO', 'VOTO NULO'])
    ].copy()
    resultados_presidente = resultados_presidente[
        ~nomes_votaveis.isin(['VOTO BRANCO', 'VOTO NULO'])
    ].copy()
    if resultados_presidente.empty:
        raise ValueError(f'Não há votos nominais para presidente em {ano_eleicao}.')

    colunas_local = {
        'NR_LOCAL_VOTACAO', 'NM_LOCAL_VOTACAO', 'DS_LOCAL_VOTACAO_ENDERECO',
        'latitude', 'longitude', 'bairro_osm',
    }
    locais_mapa = pd.read_csv(arquivo_locais_mapa, encoding='utf-8-sig')
    colunas_local_ausentes = colunas_local.difference(locais_mapa.columns)
    if colunas_local_ausentes:
        raise ValueError(f'Colunas ausentes no arquivo de locais: {sorted(colunas_local_ausentes)}')

    colunas_chave_local = ['NR_LOCAL_VOTACAO', '_nome_local_chave', '_endereco_local_chave']
    for tabela in (locais_mapa, resultados_presidente, votos_brancos_nulos):
        tabela['NR_LOCAL_VOTACAO'] = (
            tabela['NR_LOCAL_VOTACAO'].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
        )
        tabela['_nome_local_chave'] = (
            tabela['NM_LOCAL_VOTACAO'].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()
        )
        tabela['_endereco_local_chave'] = (
            tabela['DS_LOCAL_VOTACAO_ENDERECO'].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()
        )
        if tabela[colunas_chave_local].isna().any().any() or tabela[colunas_chave_local].eq('').any().any():
            raise ValueError('Há identificadores, nomes ou endereços de local vazios; não é possível associar as seções com segurança.')

    if locais_mapa.duplicated(colunas_chave_local).any():
        raise ValueError('O arquivo de locais repete o mesmo NR_LOCAL_VOTACAO, nome e endereço.')
    ids_locais_repetidos = locais_mapa.loc[
        locais_mapa['NR_LOCAL_VOTACAO'].duplicated(keep=False), 'NR_LOCAL_VOTACAO'
    ].nunique()
    if ids_locais_repetidos:
        st.warning(
            f'{ids_locais_repetidos} números de local aparecem em endereços distintos; '
            'as seções serão associadas pelo número, nome e endereço do turno correspondente.'
        )

    locais_mapa['latitude'] = pd.to_numeric(locais_mapa['latitude'], errors='coerce')
    locais_mapa['longitude'] = pd.to_numeric(locais_mapa['longitude'], errors='coerce')
    locais_mapa['bairro_osm'] = (
        locais_mapa['bairro_osm'].astype('string').str.strip().replace('', pd.NA).fillna('Bairro não identificado')
    )
    locais_mapa = locais_mapa[
        locais_mapa['latitude'].between(-90, 90)
        & locais_mapa['longitude'].between(-180, 180)
    ].copy()
    if locais_mapa.empty:
        raise ValueError('O arquivo de locais não contém coordenadas válidas.')


    chaves_secao = ['NR_ZONA', 'NR_SECAO'] + colunas_chave_local
    return (
        resultados_presidente,
        votos_brancos_nulos,
        locais_mapa,
        colunas_chave_local,
        chaves_secao,
        ano_eleicao,
    )


@st.cache_data(show_spinner='Carregando limite municipal do IBGE...')
def carregar_limite_municipal(caminho: str, assinatura: tuple[int, int]):
    limite = gpd.read_file(caminho)
    if limite.empty or limite.crs is None:
        raise ValueError('O GeoJSON do limite municipal não contém geometria ou CRS definido.')
    if 'codarea' not in limite.columns or set(limite['codarea'].astype(str)) != {'3122306'}:
        raise ValueError('O arquivo de limite não corresponde ao município de Divinópolis (IBGE 3122306).')

    limite = limite.to_crs(CRS_METRICO).geometry.union_all()
    if limite.is_empty or not limite.is_valid:
        raise ValueError('A geometria do limite municipal está vazia ou é inválida.')
    return limite


st.set_page_config(page_title='Mapa eleitoral de Divinópolis', layout='wide')
assinatura_resultados = (CAMINHO_RESULTADOS.stat().st_mtime_ns, CAMINHO_RESULTADOS.stat().st_size)
assinatura_locais = (CAMINHO_LOCAIS.stat().st_mtime_ns, CAMINHO_LOCAIS.stat().st_size)
assinatura_limite = (CAMINHO_LIMITE_MUNICIPAL.stat().st_mtime_ns, CAMINHO_LIMITE_MUNICIPAL.stat().st_size)
cache_version = (*assinatura_resultados, *assinatura_locais, *assinatura_limite)
(
    resultados_presidente,
    votos_brancos_nulos,
    locais_mapa,
    colunas_chave_local,
    chaves_secao,
    ano_eleicao,
) = carregar_dados(str(CAMINHO_RESULTADOS), str(CAMINHO_LOCAIS), assinatura_resultados, assinatura_locais)
limite_municipal = carregar_limite_municipal(str(CAMINHO_LIMITE_MUNICIPAL), assinatura_limite)
candidatos = sorted(resultados_presidente['NM_VOTAVEL'].dropna().unique())
paleta = [
    '#225EA8', '#D73027', '#1A9850', '#FDAE61', '#756BB1', '#1D91C0',
    '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666', '#A6CEE3',
    '#FB9A99', '#B2DF8A', '#CAB2D6', '#FDBF6F',
]
candidatos = sorted(resultados_presidente['NM_VOTAVEL'].dropna().unique())
def normalizar_candidato(nome):
    import unicodedata
    return ''.join(
        caractere for caractere in unicodedata.normalize('NFKD', str(nome))
        if not unicodedata.combining(caractere)
    ).upper()


def cores_por_candidato(candidatos_turno):
    cores = {}
    cores_extras = iter(paleta[2:])
    for nome in candidatos_turno:
        chave = normalizar_candidato(nome)
        if 'FLAVIO' in chave and 'BOLSONARO' in chave:
            cores[nome] = '#1E5AA8'
        elif 'LUIZ INACIO LULA' in chave:
            cores[nome] = '#D71920'
        else:
            cores[nome] = next(cores_extras, '#506784')
    return cores




def formatar_votos(valor):
    return f'{int(valor):,}'.replace(',', '.')


def formatar_percentual(valor):
    return f'{valor:.1%}'.replace('.', ',')


def fecho_convexo(pontos):
    pontos_ordenados = sorted(set(pontos))
    if len(pontos_ordenados) < 3:
        raise ValueError('São necessários ao menos três locais de votação distintos para construir o Voronoi.')

    def produto_vetorial(origem, primeiro, segundo):
        return (
            (primeiro[0] - origem[0]) * (segundo[1] - origem[1])
            - (primeiro[1] - origem[1]) * (segundo[0] - origem[0])
        )

    inferior = []
    for ponto in pontos_ordenados:
        while len(inferior) >= 2 and produto_vetorial(inferior[-2], inferior[-1], ponto) <= 0:
            inferior.pop()
        inferior.append(ponto)

    superior = []
    for ponto in reversed(pontos_ordenados):
        while len(superior) >= 2 and produto_vetorial(superior[-2], superior[-1], ponto) <= 0:
            superior.pop()
        superior.append(ponto)

    contorno = inferior[:-1] + superior[:-1]
    if len(contorno) < 3:
        raise ValueError('Os locais de votação não formam uma área bidimensional para o Voronoi.')
    return contorno


def recortar_por_meio_plano(poligono, a, b, limite):
    if not poligono:
        return []

    resultado = []
    anterior = poligono[-1]
    valor_anterior = a * anterior[0] + b * anterior[1] - limite
    dentro_anterior = valor_anterior <= 1e-7

    for atual in poligono:
        valor_atual = a * atual[0] + b * atual[1] - limite
        dentro_atual = valor_atual <= 1e-7
        if dentro_atual != dentro_anterior:
            proporcao = valor_anterior / (valor_anterior - valor_atual)
            resultado.append((
                anterior[0] + proporcao * (atual[0] - anterior[0]),
                anterior[1] + proporcao * (atual[1] - anterior[1]),
            ))
        if dentro_atual:
            resultado.append(atual)
        anterior = atual
        valor_anterior = valor_atual
        dentro_anterior = dentro_atual

    return resultado


def poligonos_voronoi(pontos_metros, limites_municipio):
    pontos = [tuple(ponto) for ponto in pontos_metros]
    fecho_convexo(pontos)
    min_x, min_y, max_x, max_y = limites_municipio
    limite_externo = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    celulas = []

    for indice, ponto in enumerate(pontos):
        celula = limite_externo.copy()
        for outro_indice, outro in enumerate(pontos):
            if indice == outro_indice:
                continue
            a = outro[0] - ponto[0]
            b = outro[1] - ponto[1]
            limite = (outro[0] ** 2 + outro[1] ** 2 - ponto[0] ** 2 - ponto[1] ** 2) / 2
            celula = recortar_por_meio_plano(celula, a, b, limite)
            if not celula:
                break
        if len(celula) >= 3:
            celulas.append(celula)
        else:
            raise ValueError('Não foi possível gerar uma célula de Voronoi para cada local de votação.')

    return celulas


@st.cache_resource(show_spinner='Gerando mapa do turno...')
def criar_mapa_turno(turno: int, cache_version: tuple[int, int, int, int, int, int]):
    resultados = resultados_presidente[resultados_presidente['NR_TURNO'] == turno]
    if resultados.empty:
        raise ValueError(f'Não há resultados presidenciais do {turno}º turno em {ano_eleicao}.')
    candidatos_turno = sorted(resultados['NM_VOTAVEL'].dropna().unique())
    if turno == 2 and len(candidatos_turno) != 2:
        raise ValueError(f'O segundo turno deveria ter dois candidatos, mas foram encontrados {len(candidatos_turno)}.')
    cores_turno = cores_por_candidato(candidatos_turno)

    totais = resultados.groupby(chaves_secao + ['NM_VOTAVEL'], as_index=False)['QT_VOTOS'].sum()
    votos_validos = (
        totais.groupby(chaves_secao, as_index=False)['QT_VOTOS'].sum()
        .rename(columns={'QT_VOTOS': 'votos_validos'})
    )
    vencedores_secao = (
        totais.sort_values(
            chaves_secao + ['QT_VOTOS', 'NM_VOTAVEL'],
            ascending=[True] * len(chaves_secao) + [False, True],
            kind='mergesort',
        )
        .drop_duplicates(chaves_secao)
        .rename(columns={'NM_VOTAVEL': 'lider_secao', 'QT_VOTOS': 'votos_lider_secao'})
    )
    secoes = votos_validos.merge(
        vencedores_secao[chaves_secao + ['lider_secao', 'votos_lider_secao']],
        on=chaves_secao,
        validate='one_to_one',
    )
    secoes['percentual_lider_secao'] = secoes['votos_lider_secao'] / secoes['votos_validos']

    dados_locais = totais.merge(
        locais_mapa[colunas_chave_local + ['latitude', 'longitude', 'NM_LOCAL_VOTACAO', 'bairro_osm']],
        on=colunas_chave_local,
        how='left',
        validate='many_to_one',
    )
    votos_por_local = (
        dados_locais.groupby(colunas_chave_local + ['latitude', 'longitude', 'NM_VOTAVEL'], as_index=False)
        ['QT_VOTOS'].sum()
    )
    vencedores_local = (
        votos_por_local.sort_values(
            colunas_chave_local + ['QT_VOTOS', 'NM_VOTAVEL'],
            ascending=[True] * len(colunas_chave_local) + [False, True],
            kind='mergesort',
        )
        .drop_duplicates(colunas_chave_local)
        .rename(columns={'NM_VOTAVEL': 'lider_local', 'QT_VOTOS': 'votos_lider_local'})
    )
    totais_local = (
        votos_por_local.groupby(colunas_chave_local, as_index=False)['QT_VOTOS'].sum()
        .rename(columns={'QT_VOTOS': 'votos_validos_local'})
    )
    vencedores_local = vencedores_local.merge(totais_local, on=colunas_chave_local, validate='one_to_one')
    vencedores_local['percentual_lider_local'] = (
        vencedores_local['votos_lider_local'] / vencedores_local['votos_validos_local']
    )

    secoes = secoes.merge(locais_mapa, on=colunas_chave_local, how='left', validate='many_to_one')
    secoes = secoes.merge(
        vencedores_local[colunas_chave_local + ['lider_local', 'votos_lider_local', 'votos_validos_local', 'percentual_lider_local']],
        on=colunas_chave_local,
        how='left',
        validate='many_to_one',
    )
    sem_coordenadas = secoes['latitude'].isna() | secoes['longitude'].isna()
    if sem_coordenadas.any():
        st.warning(f'Seções sem coordenadas válidas, não exibidas: {int(sem_coordenadas.sum())}.')
    secoes = secoes[~sem_coordenadas].copy()
    if secoes.empty:
        raise ValueError(f'Nenhuma seção do {turno}º turno possui coordenadas válidas.')

    votos_por_ponto = (
        votos_por_local.groupby(['latitude', 'longitude', 'NM_VOTAVEL'], as_index=False)['QT_VOTOS'].sum()
    )
    vencedores_ponto = (
        votos_por_ponto.sort_values(
            ['latitude', 'longitude', 'QT_VOTOS', 'NM_VOTAVEL'],
            ascending=[True, True, False, True],
            kind='mergesort',
        )
        .drop_duplicates(['latitude', 'longitude'])
        .rename(columns={'NM_VOTAVEL': 'lider_ponto', 'QT_VOTOS': 'votos_lider_ponto'})
    )
    totais_ponto = (
        votos_por_ponto.groupby(['latitude', 'longitude'], as_index=False)['QT_VOTOS'].sum()
        .rename(columns={'QT_VOTOS': 'votos_validos_ponto'})
    )
    vencedores_ponto = vencedores_ponto.merge(
        totais_ponto, on=['latitude', 'longitude'], validate='one_to_one'
    )
    vencedores_ponto['percentual_lider_ponto'] = (
        vencedores_ponto['votos_lider_ponto'] / vencedores_ponto['votos_validos_ponto']
    )
    locais_por_ponto = (
        dados_locais.groupby(['latitude', 'longitude'], as_index=False)
        .agg(
            locais=('NM_LOCAL_VOTACAO', lambda nomes: ' / '.join(sorted(set(nomes.astype(str))))),
            bairro_osm=('bairro_osm', lambda bairros: ' / '.join(sorted(set(bairros.dropna().astype(str))))),
        )
    )
    vencedores_ponto = vencedores_ponto.merge(
        locais_por_ponto, on=['latitude', 'longitude'], validate='one_to_one'
    )

    latitude_centro = float(vencedores_ponto['latitude'].mean())
    longitude_centro = float(vencedores_ponto['longitude'].mean())
    pontos_geograficos = gpd.GeoSeries(
        gpd.points_from_xy(vencedores_ponto['longitude'], vencedores_ponto['latitude']),
        crs='EPSG:4326',
    )
    pontos_metros = [
        (ponto.x, ponto.y)
        for ponto in pontos_geograficos.to_crs(CRS_METRICO)
    ]
    celulas_voronoi = poligonos_voronoi(pontos_metros, limite_municipal.bounds)
    propriedades_voronoi = []
    for indice in range(len(celulas_voronoi)):
        registro = vencedores_ponto.iloc[indice]
        bairros = registro['bairro_osm'] or 'Não identificado'
        propriedades_voronoi.append({
            'local': registro['locais'],
            'bairros': bairros,
            'lider': registro['lider_ponto'],
            'votos': formatar_votos(registro['votos_lider_ponto']),
            'percentual': formatar_percentual(registro['percentual_lider_ponto']),
            'cor': cores_turno[registro['lider_ponto']],
        })
    areas_voronoi = gpd.GeoDataFrame(
        propriedades_voronoi,
        geometry=[Polygon(celula) for celula in celulas_voronoi],
        crs=CRS_METRICO,
    )
    areas_voronoi.geometry = areas_voronoi.geometry.intersection(limite_municipal)
    areas_voronoi = areas_voronoi[
        areas_voronoi.geometry.notna()
        & ~areas_voronoi.geometry.is_empty
        & areas_voronoi.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    ]
    if areas_voronoi.empty:
        raise ValueError('Nenhuma área de Voronoi intersecta o limite municipal do IBGE.')
    features_voronoi = json.loads(areas_voronoi.to_crs('EPSG:4326').to_json())['features']
    limite_municipal_wgs84 = gpd.GeoSeries([limite_municipal], crs=CRS_METRICO).to_crs('EPSG:4326').iloc[0]

    mapa = folium.Map(
        location=[latitude_centro, longitude_centro],
        zoom_start=12,
        tiles='OpenStreetMap',
        control_scale=True,
    )
    folium.GeoJson(
        {'type': 'FeatureCollection', 'features': features_voronoi},
        name='Áreas aproximadas de Voronoi',
        style_function=lambda feature: {
            'fillColor': feature['properties']['cor'],
            'color': '#FFFFFF',
            'weight': 1.2,
            'fillOpacity': 0.48,
        },
        highlight_function=lambda feature: {'weight': 3, 'fillOpacity': 0.68},
        tooltip=folium.GeoJsonTooltip(
            fields=['local', 'bairros', 'lider', 'votos', 'percentual'],
            aliases=['Local(is) de votação:', 'Bairro(s) do local (referência):', 'Candidato líder:', 'Votos no local:', 'Percentual dos válidos:'],
            localize=True,
            sticky=True,
            labels=True,
            class_name='votacao-tooltip',
        ),
    ).add_to(mapa)
    folium.GeoJson(
        limite_municipal_wgs84.__geo_interface__,
        name='Limite municipal oficial (IBGE)',
        style_function=lambda feature: {
            'color': '#202124',
            'weight': 2.5,
            'fillOpacity': 0,
        },
    ).add_to(mapa)
    min_longitude, min_latitude, max_longitude, max_latitude = limite_municipal_wgs84.bounds
    mapa.fit_bounds(
        [
            [min_latitude, min_longitude],
            [max_latitude, max_longitude],
        ],
        padding=[24, 24],
    )
    agrupamento = MarkerCluster(
        name='Seções eleitorais', show_coverage_on_hover=False, spiderfy_on_max_zoom=True
    ).add_to(mapa)
    votos_por_secao = (
        totais.pivot(index=chaves_secao, columns='NM_VOTAVEL', values='QT_VOTOS')
        .reindex(columns=candidatos_turno, fill_value=0)
        .fillna(0)
    )
    votos_brancos_nulos_por_secao = votos_brancos_nulos.groupby(
        chaves_secao + ['NR_TURNO', 'NM_VOTAVEL']
    )['QT_VOTOS'].sum()

    for _, secao in secoes.iterrows():
        chave = tuple(secao[coluna] for coluna in chaves_secao)
        votos_secao = votos_por_secao.loc[chave]
        linhas_brancos_nulos = ''.join(
            '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                categoria,
                formatar_votos(votos_brancos_nulos_por_secao.get((*chave, 1, categoria), 0)),
                formatar_votos(votos_brancos_nulos_por_secao.get((*chave, 2, categoria), 0)),
            )
            for categoria in ('VOTO BRANCO', 'VOTO NULO')
        )
        linhas_votos = ''.join(
            '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                escape(str(candidato)),
                formatar_votos(votos_secao[candidato]),
                formatar_percentual(votos_secao[candidato] / secao['votos_validos']),
            )
            for candidato in sorted(candidatos_turno, key=lambda nome: (-votos_secao[nome], nome))
        )
        cor = cores_turno[secao['lider_secao']]
        tooltip_html = (
            '<div class="votacao-card">'
            f'<div class="votacao-bairro">{escape(str(secao["bairro_osm"]))}</div>'
            f'<div class="votacao-local">{escape(str(secao["NM_LOCAL_VOTACAO"]))}</div>'
            f'<div class="votacao-secao">Zona {int(secao["NR_ZONA"])} · Seção {int(secao["NR_SECAO"])}</div>'
            f'<div class="votacao-lider">Líder na seção: <strong>{escape(str(secao["lider_secao"]))}</strong> '
            f'({formatar_percentual(secao["percentual_lider_secao"])}; '
            f'{formatar_votos(secao["votos_lider_secao"])} votos)</div>'
            f'<div class="votacao-regiao">Líder no local: {escape(str(secao["lider_local"]))} '
            f'({formatar_percentual(secao["percentual_lider_local"])})</div>'
            '<div class="votacao-secao">Votos brancos e nulos</div>'
            '<table><thead><tr><th>Tipo</th><th>1º turno</th><th>2º turno</th></tr></thead>'
            f'<tbody>{linhas_brancos_nulos}</tbody></table>'
            '<table><thead><tr><th>Candidato</th><th>Votos</th><th>% válidos</th></tr></thead>'
            f'<tbody>{linhas_votos}</tbody></table>'
            '</div>'
        )
        folium.Marker(
            location=[secao['latitude'], secao['longitude']],
            icon=folium.DivIcon(
                icon_size=(16, 16), icon_anchor=(8, 8),
                html=f'<div style="width:14px;height:14px;border:2px solid #fff;border-radius:50%;background:{cor};box-shadow:0 1px 4px #333;"></div>',
            ),
            popup=folium.Popup(
                tooltip_html, max_width=420, max_height=380
            ),
        ).add_to(agrupamento)

    contagem_areas = vencedores_ponto['lider_ponto'].value_counts().to_dict()
    itens_legenda = ''.join(
        f'<div class="legenda-item"><span class="legenda-cor" style="background:{cores_turno[candidato]}"></span>'
        f'<span>{escape(str(candidato))}</span><b>{int(contagem_areas.get(candidato, 0))}</b></div>'
        for candidato in candidatos_turno
    )
    estilo = """
    <style>
      .votacao-tooltip { background: #fff; border: 0; border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,.24); padding: 8px 10px; font: 12px/1.4 Arial,sans-serif; }
      .votacao-tooltip:before { display: none; }
      .votacao-card { min-width: 290px; max-width: 390px; max-height: 340px; overflow-y: auto; overscroll-behavior: contain; padding: 10px 12px; border-radius: 8px; background: #fff; color: #202124; box-shadow: 0 2px 12px rgba(0,0,0,.28); font: 12px/1.4 Arial,sans-serif; }
      .votacao-bairro { font-size: 15px; font-weight: 700; }
      .votacao-local { color: #555; margin: 2px 0; }
      .votacao-secao { font-weight: 700; margin: 6px 0; }
      .votacao-lider { margin: 5px 0; }
      .votacao-card table { width: 100%; border-collapse: collapse; margin-top: 6px; }
      .votacao-card th, .votacao-card td { padding: 3px 4px; border-bottom: 1px solid #e5e7eb; text-align: left; }
      .votacao-card th:nth-child(n+2), .votacao-card td:nth-child(n+2) { text-align: right; }
      .votacao-regiao { margin-top: 6px; color: #555; font-size: 11px; }
      .mapa-legenda { position: fixed; z-index: 9999; top: 12px; right: 12px; max-width: 310px; max-height: 45vh; overflow-y: auto; padding: 12px; border: 1px solid #6B7280; border-radius: 8px; background: #FFFFFF; color: #111827 !important; box-shadow: 0 2px 12px rgba(0,0,0,.35); font: 13px/1.5 Arial,sans-serif; }
      .mapa-legenda * { color: #111827 !important; opacity: 1 !important; }
      .mapa-legenda h4 { margin: 0 0 4px; font-size: 15px; font-weight: 700; }
      .legenda-subtitulo { margin-bottom: 8px; color: #374151 !important; font-size: 11px; font-weight: 600; }
      .legenda-item { display: grid; grid-template-columns: 14px 1fr auto; gap: 8px; align-items: center; margin: 5px 0; }
      .legenda-cor { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #777; }
      .legenda-item b { min-width: 24px; padding: 1px 6px; border: 1px solid #9CA3AF; border-radius: 999px; background: #E5E7EB; text-align: center; font-weight: 700; }
      .legenda-nota { margin-top: 8px; color: #374151 !important; font-size: 11px; }
    </style>
    """
    legenda = (
        f'<div class="mapa-legenda"><h4>Presidente · {turno}º turno · {ano_eleicao}</h4>'
        '<div class="legenda-subtitulo">Áreas de Voronoi lideradas por candidato</div>'
        f'{itens_legenda}<div class="legenda-nota">Cada polígono representa a área aproximada mais próxima '
        'de um local de votação; a cor indica o candidato com mais votos somados nas seções desse local. '
        'As células de Voronoi foram recortadas ao limite municipal oficial do IBGE; não representam limites de bairros. '
        'Os pontos mostram as seções eleitorais. Clique nos agrupamentos para ampliá-los.</div></div>'
    )
    mapa.get_root().html.add_child(folium.Element(estilo + legenda))
    folium.LayerControl(collapsed=True).add_to(mapa)
    return mapa



st.title(f'Resultados presidenciais de Divinópolis - {ano_eleicao}')
st.caption('As áreas de Voronoi são aproximadas e recortadas ao limite municipal oficial do IBGE; não são limites de bairros.')
turno = st.radio('Turno', options=[1, 2], format_func=lambda valor: f'{valor}º turno', horizontal=True)
mapa = criar_mapa_turno(turno, cache_version)
st_folium(mapa, height=720, width=1200, returned_objects=[])