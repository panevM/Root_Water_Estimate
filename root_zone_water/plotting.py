"""Plotly графици од веќе пресметани влезови и резултати на runner.

Тетратките користат area/inputs и графици за состојби, иновации и чувствителност.
Функциите враќаат Figure или речник од Figure, без преземање и без повторно
проценување. Само export_html запишува датотека; прикажувањето го прави
повикувачот. language='mk' ги избира постојните македонски текстови каде
се поддржани, а другите вредности ги избираат англиските.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FILTER_COLORS = {'open_loop': '#1f77b4', 'ekf': '#d62728', 'ukf': '#2ca02c'}
FILTER_STYLES = {
    'open_loop': {'posterior': '#1f77b4', 'prior': '#7fa6c9', 'band': 'rgba(31,119,180,0.10)'},
    'ekf': {'posterior': '#d62728', 'prior': '#e88989', 'band': 'rgba(214,39,40,0.10)'},
    'ukf': {'posterior': '#2ca02c', 'prior': '#82c982', 'band': 'rgba(44,160,44,0.10)'},
}
PLOTLY_CONFIG = {'scrollZoom': True, 'responsive': True, 'displaylogo': False}

def _labels(language):
    """Врати речник на постојните ознаки за графици според language ('mk' или друго)."""
    if language != 'mk':
        return {'area': 'Configured demonstration area near Skopje', 'weather': 'Weather forcing',
                'ndvi': 'NDVI observations', 'ndmi': 'NDMI observations', 'kc': 'Crop coefficient',
                'inputs': 'Inputs and accepted satellite observations', 'prior': 'prior', 'posterior': 'posterior',
                'band': 'conditional ±2σ', 'proxy': 'NDMI-derived storage proxy (speculative)',
                'state': 'Storage W (mm)', 'date': 'Date (UTC)', 'innovation': 'Innovation (mm)',
                'standardized': 'Innovation / √S', 'post_storage': 'Posterior W (mm)', 'std': 'Conditional σ (mm)'}
    return {'area': 'Конфигурирана демонстрациска област во близина на Скопје', 'weather': 'Метеоролошки влезни податоци',
            'ndvi': 'Набљудувања на NDVI', 'ndmi': 'Набљудувања на NDMI', 'kc': 'Коефициент на културата',
            'inputs': 'Влезни податоци и прифатени сателитски набљудувања', 'prior': 'претходна процена',
            'posterior': 'корегирана процена', 'band': 'условна ±2σ',
            'proxy': 'Индиректна процена од NDMI (спекулативна)', 'state': 'Количество вода W (mm)',
            'date': 'Датум (UTC)', 'innovation': 'Иновација (mm)', 'standardized': 'Иновација / √S',
            'post_storage': 'Корегирана процена W (mm)', 'std': 'Условна σ (mm)'}

def _date(values):
    """Претвори низа датуми/ISO времиња во pandas датуми за оските, без промена на влезот."""
    return pd.to_datetime(values)

def _base_layout(fig, title, height):
    """Постави заеднички изглед, title и height во пиксели; измени и врати ја fig."""
    fig.update_layout(title=title, height=height, template='plotly_white', autosize=True,
                      hovermode='x unified', legend=dict(groupclick='togglegroup'),
                      margin=dict(l=70, r=30, t=70, b=55))
    fig.update_xaxes(showgrid=True, rangeslider_visible=False, showspikes=True,
                     spikemode='across', spikesnap='cursor')
    return fig

def area(c, satellite, language='en'):
    """Врати график на надворешниот прстен од c['geometry'] во географски степени.

    satellite е табелата од load_data; прифатениот vegetation дава медијански
    SCL удел како контекст, без идентификација на културата. language избира
    текстови; влезовите се читаат и нема преземање карта.
    """
    labels = _labels(language)
    ring = np.array(c['geometry']['coordinates'][0])
    fig = go.Figure(go.Scatter(x=ring[:, 0], y=ring[:, 1], mode='lines+markers',
                               name='Configured polygon', fill='toself',
                               fillcolor='rgba(0,128,128,.12)', line=dict(color='teal')))
    context = 'Нема сателитски контекст за покриеноста; соодветноста на областа е нерешена.' if language == 'mk' else 'No satellite land-cover context captured; field suitability unresolved.'
    if 'vegetation' in satellite and satellite['vegetation_accepted'].any():
        v = satellite.loc[satellite['vegetation_accepted'], 'vegetation'].median()
        context = (f'Медијански удел на вегетација SCL кај прифатените пиксели: {v:.1%}. SCL не може да идентификува пченица.' if language == 'mk' else f'Median SCL vegetation share among accepted pixels: {v:.1%}. SCL cannot identify wheat.')
        if v < .5: context += ' Ниска вегетација: несоодветно за репрезентативна парцела.' if language == 'mk' else ' Low vegetation: unsuitable as a representative crop parcel.'
    fig.add_annotation(xref='paper', yref='paper', x=0, y=-.18, showarrow=False, text=context, align='left')
    fig.update_layout(title=labels['area'], height=500, autosize=True,
                      template='plotly_white', margin=dict(l=70, r=30, t=70, b=100))
    fig.update_xaxes(title='Географска должина (степени источно)' if language == 'mk' else 'Longitude (degrees east)', scaleanchor='y')
    fig.update_yaxes(title='Географска ширина (степени северно)' if language == 'mk' else 'Latitude (degrees north)')
    return fig

def inputs(weather, satellite, result, language='en'):
    """Врати четири панели: дневни врнежи/ET0, прифатени NDVI/NDMI и избраниот Kc.

    weather/satellite се табелите од load_data, а result е усогласен дневен
    резултат од run. Водните влезови се mm/ден, индексите и Kc се
    бездимензионални. language избира текстови; не се пополнуваат отсутни снимки.
    """
    labels = _labels(language)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.05,
                        subplot_titles=(labels['weather'], labels['ndvi'], labels['ndmi'], labels['kc']))
    t = _date(weather['date'])
    fig.add_trace(go.Bar(x=t, y=weather['precipitation_mm'], name='Врнежи (mm/ден)' if language == 'mk' else 'Precipitation (mm/day)', marker_color='#4c78a8'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=weather['et0_mm'], name='Референтен ET0 од Open-Meteo (mm/ден)' if language == 'mk' else 'Open-Meteo reference ET0 (mm/day)', line=dict(color='#f58518')), row=1, col=1)
    for row, index in ((2, 'ndvi'), (3, 'ndmi')):
        if len(satellite):
            valid = satellite[satellite[index + '_accepted']]
            fig.add_trace(go.Scatter(x=_date(valid['date']), y=valid[index], mode='markers',
                                     name=('Прифатен реален ' if language == 'mk' else 'Accepted real ') + index.upper(), marker=dict(size=8, color='#54a24b')), row=row, col=1)
        if not len(satellite) or not satellite[index + '_accepted'].any():
            fig.add_annotation(x=.5, y=.5, xref=f'x{row} domain', yref=f'y{row} domain',
                               text=('Нема преземени набљудувања за ' if language == 'mk' else 'No acquired ') + index.upper() + (' набљудувања' if language == 'mk' else ' observations'), showarrow=False)
        fig.update_yaxes(title_text=index.upper() + ' (без димензија)', range=[-1, 1], row=row, col=1)
    fig.add_trace(go.Scatter(x=t, y=result['kc'], name='Избран Kc' if language == 'mk' else 'Selected Kc', line=dict(color='#59a14f')), row=4, col=1)
    fig.update_yaxes(title_text='mm/ден' if language == 'mk' else 'mm/day', row=1, col=1); fig.update_yaxes(title_text='Kc', row=4, col=1)
    fig.update_xaxes(title_text=labels['date'], row=4, col=1)
    return _base_layout(fig, labels['inputs'], 950)

def states(results, p, language='en'):
    """Врати поврзани панели од непразен речник results: име на филтер→DataFrame.

    Тетратките и state_figures предаваат резултати од run и Parameters p
    за линиите wp, fc−RAW, fc и заситеност. На крајот на секој UTC ден се
    прикажуваат предвидена процена W_prior_mm и корегирана процена
    W_posterior_mm, со лента ±2√P_posterior_mm2 во mm. Лентата не се
    отсекува на физичките граници и е условна на моделските претпоставки.
    Маркерите се индиректна процена на количеството вода добиена од NDMI,
    не директни теренски мерења. Дополнителните податоци при посочување
    вклучуваат иновација, Калманово засилување и √Q_t. language избира текстови.
    """
    labels = _labels(language)
    fig = make_subplots(rows=len(results), cols=1, shared_xaxes=True, vertical_spacing=.045,
                        subplot_titles=[name.upper() + (' — процена на количеството вода од претпоставките' if language == 'mk' else ' — speculative storage conditional on assumptions') for name in results])
    for row, (name, result) in enumerate(results.items(), 1):
        style = FILTER_STYLES.get(name, {'posterior': '#636363', 'prior': '#a0a0a0', 'band': 'rgba(99,99,99,0.08)'})
        t = _date(result['interval_end_utc'])
        prior_sd = np.sqrt(result['P_prior_mm2']); posterior_sd = np.sqrt(result['P_posterior_mm2'])
        prior_hover = np.column_stack([result['W_prior_mm'], prior_sd])
        post_hover = np.column_stack([result['W_posterior_mm'], posterior_sd, result['innovation_mm'].fillna(np.nan),
                                      result['gain'].fillna(np.nan), np.sqrt(result['measurement_covariance_Qt_mm2'].fillna(np.nan))])
        fig.add_trace(go.Scatter(x=t, y=result['W_posterior_mm'] + 2*posterior_sd, name=f'{name.upper()} +2σ',
                                 legendgroup=name, line=dict(width=0), hoverinfo='skip', showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=result['W_posterior_mm'] - 2*posterior_sd, name=f'{name.upper()} {labels["band"]}',
                                 legendgroup=name, line=dict(width=0), fill='tonexty', fillcolor=style['band'], hoverinfo='skip'), row=row, col=1)
        hover_prior = ('Датум: %{x|%Y-%m-%d}<br>Филтер: ' if language == 'mk' else 'Date: %{x|%Y-%m-%d}<br>Filter: ')+name.upper()+('<br>Претходна процена: ' if language == 'mk' else '<br>Prior storage: ')+'%{customdata[0]:.2f} mm<br>'+('Претходна σ: ' if language == 'mk' else 'Prior σ: ')+'%{customdata[1]:.2f} mm<extra></extra>'
        hover_post = ('Датум: %{x|%Y-%m-%d}<br>Филтер: ' if language == 'mk' else 'Date: %{x|%Y-%m-%d}<br>Filter: ')+name.upper()+('<br>Корегирана процена: ' if language == 'mk' else '<br>Posterior storage: ')+'%{customdata[0]:.2f} mm<br>'+('Стандардна девијација на состојбата: ' if language == 'mk' else 'State σ: ')+'%{customdata[1]:.2f} mm<br>'+('Иновација: ' if language == 'mk' else 'Innovation: ')+'%{customdata[2]:.2f} mm<br>'+('Калманов засилувач: ' if language == 'mk' else 'Kalman gain: ')+'%{customdata[3]:.3f}<br>'+('Мерна σ: ' if language == 'mk' else 'Measurement σ: ')+'%{customdata[4]:.2f} mm<extra></extra>'
        fig.add_trace(go.Scatter(x=t, y=result['W_prior_mm'], mode='lines', name=f'{name.upper()} {labels["prior"]}', legendgroup=name,
                                 line=dict(color=style['prior'], dash='dash', width=2), customdata=prior_hover,
                                 hovertemplate=hover_prior), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=result['W_posterior_mm'], mode='lines', name=f'{name.upper()} {labels["posterior"]}', legendgroup=name,
                                 line=dict(color=style['posterior'], width=3), customdata=post_hover,
                                 hovertemplate=hover_post), row=row, col=1)
        obs = result[result['proxy_mm'].notna()]
        if len(obs):
            obs_hover = np.column_stack([obs['proxy_mm'], obs['innovation_mm'].fillna(np.nan), obs['gain'].fillna(np.nan),
                                          np.sqrt(obs['measurement_covariance_Qt_mm2'].fillna(np.nan))])
            fig.add_trace(go.Scatter(x=_date(obs['interval_end_utc']), y=obs['proxy_mm'], mode='markers',
                                     name=labels['proxy'], legendgroup='proxy',
                                     marker=dict(color='#111111', symbol='x', size=9), customdata=obs_hover,
                                     hovertemplate=('Датум: %{x|%Y-%m-%d}<br>Филтер: ' if language == 'mk' else 'Date: %{x|%Y-%m-%d}<br>Filter: ')+name.upper()+('<br>Индиректна процена од NDMI: ' if language == 'mk' else '<br>Speculative storage proxy: ')+'%{y:.2f} mm<br>'+('Иновација: ' if language == 'mk' else 'Innovation: ')+'%{customdata[1]:.2f} mm<br>'+('Калманов засилувач: ' if language == 'mk' else 'Kalman gain: ')+'%{customdata[2]:.3f}<br>'+('Мерна σ: ' if language == 'mk' else 'Measurement σ: ')+'%{customdata[3]:.2f} mm<extra></extra>'), row=row, col=1)
        labels_ref = ('Точка на венеење', 'Почеток на стрес', 'Полски капацитет', 'Претпоставена заситеност') if language == 'mk' else ('Wilting point', 'Stress onset', 'Field capacity', 'Assumed saturation')
        for y, label in zip((p.wp, p.fc-p.raw, p.fc, p.saturation), labels_ref):
            fig.add_hline(y=y, line=dict(color='#777777', width=1), annotation_text=label, annotation_position='bottom right', row=row, col=1)
        fig.update_yaxes(title_text=labels['state'], row=row, col=1)
    fig.update_xaxes(title_text=labels['date'], row=len(results), col=1)
    fig.add_annotation(xref='paper', yref='paper', x=0, y=-.27, showarrow=False,
                       text=('Засенчените ленти: неизвесност на проценетата состојба при зададените претпоставки (±2σ).' if language == 'mk' else 'Shaded bands: conditional state uncertainty (±2σ). NDMI markers: speculative storage proxies, not direct soil-water measurements or ground truth.'), align='left')
    fig = _base_layout(fig, 'Споредба на филтри: количество во кореновата зона' if language == 'mk' else 'Filter comparison: root-zone storage', 360*len(results)+120)
    fig.update_layout(margin=dict(l=70, r=30, t=70, b=135))
    return fig

def state_figures(results, p, language='en'):
    """Врати име→самостоен Figure преку states за results од compare и Parameters p.

    Тетратките го користат за одделно прикажување/извоз; language се пренесува
    до states, каде се објаснети процените и лентите во mm.
    """
    return {name: states({name: result}, p, language=language) for name, result in results.items()}

def innovations(results):
    """Врати два панели на иновации од results (име→резултат од run).

    Го прескокнува open_loop и деновите без иновација. Првиот панел е
    innovation_mm со ±2√S во mm; вториот е бездимензионалната иновација/√S.
    Ова е согласност со индиректната процена од NDMI пред корекција.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=('Pre-correction innovation with ±2√S', 'Standardized innovation'))
    for name, result in results.items():
        if name == 'open_loop': continue
        valid = result[result['innovation_mm'].notna()]
        if len(valid):
            t = _date(valid['interval_end_utc']); color = FILTER_COLORS.get(name)
            fig.add_trace(go.Scatter(x=t, y=valid['innovation_mm'], mode='markers', name=name.upper(),
                                     error_y=dict(type='data', array=2*np.sqrt(valid['S_mm2']), visible=True), marker=dict(color=color),
                                     hovertemplate='Date: %{x|%Y-%m-%d}<br>Filter: '+name.upper()+'<br>Innovation: %{y:.2f} mm<extra></extra>'), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=valid['innovation_mm']/np.sqrt(valid['S_mm2']), mode='markers', name=name.upper()+' standardized',
                                     legendgroup=name, showlegend=False, marker=dict(color=color)), row=2, col=1)
    for row in (1, 2): fig.add_hline(y=0, line=dict(color='#777777', width=1), row=row, col=1)
    fig.update_yaxes(title_text='Innovation (mm)', row=1, col=1); fig.update_yaxes(title_text='Innovation / √S', row=2, col=1)
    fig.update_xaxes(title_text='Date (UTC)', row=2, col=1)
    return _base_layout(fig, 'Filter innovations', 620)

def innovation_figures(results, language='en'):
    """Врати pre_correction и standardized Figure од речникот results на филтри.

    Тетратките добиваат одделни прикази на иновацијата со ±2√S (mm) и
    иновација/√S (бездимензионална), без open_loop и без NaN денови.
    language избира текстови. Тековниот код бара барем една достапна
    иновација надвор од open_loop за да ги постави насловите во циклусот;
    ако нема ниту една, локалните title/ylabel остануваат недефинирани.
    """
    figures = {}
    for key, standardized in (('pre_correction', False), ('standardized', True)):
        fig = go.Figure()
        for name, result in results.items():
            if name == 'open_loop': continue
            valid = result[result['innovation_mm'].notna()]
            if not len(valid): continue
            t = _date(valid['interval_end_utc']); color = FILTER_COLORS.get(name)
            if standardized:
                y = valid['innovation_mm'] / np.sqrt(valid['S_mm2'])
                error_y = None; title = 'Стандардизирана иновација' if language == 'mk' else 'Standardized innovation'; ylabel = 'Иновација / √S' if language == 'mk' else 'Innovation / √S'
            else:
                y = valid['innovation_mm']; error_y = dict(type='data', array=2*np.sqrt(valid['S_mm2']), visible=True)
                title = 'Пред-корекциска иновација со ±2√S' if language == 'mk' else 'Pre-correction innovation with ±2√S'; ylabel = 'Иновација (mm)' if language == 'mk' else 'Innovation (mm)'
            fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name=name.upper(), marker=dict(color=color),
                                     error_y=error_y,
                                     hovertemplate=('Датум: ' if language == 'mk' else 'Date: ')+'%{x|%Y-%m-%d}<br>'+('Филтер: ' if language == 'mk' else 'Filter: ')+name.upper()+'<br>'+('Вредност: ' if language == 'mk' else 'Value: ')+'%{y:.2f}<extra></extra>'))
        fig.add_hline(y=0, line=dict(color='#777777', width=1))
        fig.update_yaxes(title_text=ylabel); fig.update_xaxes(title_text='Date (UTC)')
        figures[key] = _base_layout(fig, title, 380)
    return figures

def sensitivities(runs):
    """Врати два панели од runs: ознака→DataFrame добиени со runner.sensitivity.

    Прикажува корегирана процена W (mm) и условна стандардна девијација
    √P (mm) по date. Ознаките на сценаријата се пренесуваат непроменети;
    нивното постојно обратно означување на Q/R е објаснето во sensitivity.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=('Posterior storage', 'Conditional state uncertainty'))
    for label, result in runs.items():
        t = _date(result['date']); fig.add_trace(go.Scatter(x=t, y=result['W_posterior_mm'], mode='lines', name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=np.sqrt(result['P_posterior_mm2']), mode='lines', name=label+' σ', legendgroup=label, showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text='Posterior W (mm)', row=1, col=1); fig.update_yaxes(title_text='Conditional σ (mm)', row=2, col=1)
    fig.update_xaxes(title_text='Date (UTC)', row=2, col=1)
    fig.add_annotation(xref='paper', yref='paper', x=0, y=-.1, showarrow=False, text='Sensitivity on identical acquired inputs; R/endpoints have no effect without observations.', align='left')
    return _base_layout(fig, 'Sensitivity comparisons', 700)

def sensitivity_figures(runs, language='en'):
    """Врати posterior_storage и conditional_std Figure за runs од sensitivity.

    Секое сценарио дава линија за корегираната процена или √P, двете во mm,
    врз истите влезни податоци. language ги избира насловите; нема ново
    извршување на филтрите или промена на нивните резултати.
    """
    figures = {}
    for key, uncertainty in (('posterior_storage', False), ('conditional_std', True)):
        fig = go.Figure()
        for label, result in runs.items():
            t = _date(result['date'])
            y = np.sqrt(result['P_posterior_mm2']) if uncertainty else result['W_posterior_mm']
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=label))
        fig.update_yaxes(title_text=('Условна σ (mm)' if uncertainty else 'Корегирана процена W (mm)') if language == 'mk' else ('Conditional σ (mm)' if uncertainty else 'Posterior W (mm)'))
        fig.update_xaxes(title_text='Датум (UTC)' if language == 'mk' else 'Date (UTC)')
        figures[key] = _base_layout(fig, ('Чувствителност: неизвесност на проценетата состојба при зададените претпоставки' if uncertainty else 'Чувствителност: корегирано количество') if language == 'mk' else ('Sensitivity: conditional state uncertainty' if uncertainty else 'Sensitivity: posterior storage'), 380)
    return figures

def export_html(fig, path):
    """Запиши fig во HTML на path и врати ја излезната Path патека.

    Тетратките го користат за споделување: создава родителски директориуми,
    заменува постојна датотека и го вградува Plotly JavaScript за локално
    отворање. Не отвора прелистувач; грешките при запис се пренесуваат.
    """
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(destination, include_plotlyjs=True, full_html=True, auto_open=False, config=PLOTLY_CONFIG)
    return destination
