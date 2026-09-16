"""Дневен воден биланс и коефициенти за преминот g во runner.run.

Скаларната состојба x_t=W_t е вкупно количество вода во една коренова зона,
изразено како висина на воден слој во mm (1 mm = 1 литар на m²). Така
состојбата и дневните водни текови се собираат без да се задава површина.
Слојот има една фиксна ефективна длабочина и просечна волуменска влажност;
нема посебни почвени слоеви, раст на коренот или просторна распределба.

Parameters ги претвора почвените влажности во количества вода; budget
ги пресметува тековите, а water_balance_jacobian го дава G_t=∂g/∂W за EKF.
Нелинеарноста по W доаѓа од праговите за стрес, ограничувањето на
евапотранспирацијата, прелевањето и дренажата. Временска промена на однапред
зададен Kc сама по себе не значи нелинеарност по состојбата.
"""
from dataclasses import dataclass
from datetime import date
import math
import numpy as np

@dataclass(frozen=True)
class Parameters:
    """Непроменливи почвени и растителни параметри од config.parameters.

    root_depth_m е фиксна ефективна длабочина во m. theta_wp, theta_fc и
    theta_sat се волуменски содржини на вода во m³/m³: точка на венеење,
    полски капацитет и заситеност, со 0 < theta_wp < theta_fc < theta_sat ≤ 1.
    Полскиот капацитет ја означува водата задржана по слободното истекување;
    во моделот дренажа има над тој праг. Под точката на венеење моделот
    не дозволува растението да извлекува вода, иако вкупното W може да е помало.

    p во (0,1) е делот од TAW што може да се исцрпи пред воден стрес;
    drainage_fraction во [0,1] е делот од вишокот над полскиот капацитет
    што истекува во еден дневен чекор. effectiveness во [0,1] е делот од
    врнежите што влегува во билансот, пред проверката за прелевање.
    """
    root_depth_m: float
    theta_fc: float
    theta_wp: float
    theta_sat: float
    p: float
    drainage_fraction: float
    effectiveness: float = 1.0

    def __post_init__(self):
        """При создавање провери конечни вредности и физички редослед на праговите.

        dataclass го повикува автоматски; не менува параметри и крева
        ValueError за недозволена длабочина, влажност или дел.
        """
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Nonfinite parameter')
        if not (self.root_depth_m > 0 and 0 < self.theta_wp < self.theta_fc < self.theta_sat <= 1
                and 0 < self.p < 1 and 0 <= self.drainage_fraction <= 1 and 0 <= self.effectiveness <= 1):
            raise ValueError('Invalid soil/crop parameters')

    @property
    def wp(self):
        """Количество вода при точка на венеење (mm): 1000·длабочина·theta_wp."""
        return 1000 * self.root_depth_m * self.theta_wp
    @property
    def fc(self):
        """Количество вода при полски капацитет (mm), со иста фиксна длабочина."""
        return 1000 * self.root_depth_m * self.theta_fc
    @property
    def saturation(self):
        """Количество вода при заситеност (mm); горна граница за моделот и филтрите."""
        return 1000 * self.root_depth_m * self.theta_sat
    @property
    def taw(self):
        """TAW: вкупна достапна вода за растението меѓу wp и fc, во mm."""
        return self.fc - self.wp
    @property
    def raw(self):
        """RAW=p·TAW (mm): лесно достапна вода; стрес почнува под fc−RAW."""
        return self.p * self.taw

def available_water(w, p):
    """Врати достапна вода clip(w−wp, 0, TAW) во mm за w (mm) и Parameters p.

    Помошна физичка величина, проверена во тестовите; не е посебна состојба
    на филтерот. Под wp е нула, а над fc останува TAW.
    """
    return float(np.clip(w - p.wp, 0, p.taw))

def depletion(w, p):
    """Врати исцрпеност clip(fc−w, 0, TAW) во mm за w (mm) и Parameters p.

    Го мери недостигот до fc: нула над fc и TAW под wp. Заедно со
    available_water дава TAW; не менува состојба.
    """
    return float(np.clip(p.fc - w, 0, p.taw))

def stress(w, p):
    """Врати бездимензионален коефициент на воден стрес Ks во [0,1].

    budget го повикува со w во mm по дождот и прелевањето, и Parameters p.
    Ks=0 при w≤wp, расте линеарно до 1 меѓу wp и fc−RAW, а потоа е 1.
    Ks ја намалува побарувачката поради недостиг на вода; Kc одделно го
    претставува влијанието на културата и нејзината развојна фаза.
    """
    return float(np.clip((w - p.wp) / ((1 - p.p) * p.taw), 0, 1))

def budget(w, rain_mm, et0_mm, kc, p):
    """Пресметај еден дневен премин g и физичките водни текови, без промена на p.

    runner.run го повикува за билансот кај тековната средна вредност и
    преку transition за EKF/UKF. w е почетното вкупно количество вода во mm,
    во [0, p.saturation]. rain_mm се вкупни дневни врнежи во mm во проверен
    период без снег; et0_mm е референтна дневна евапотранспирација во mm.
    kc е бездимензионален Kc од календар или претходно прифатен NDVI;
    p е Parameters. Влезните броеви мора да се конечни, а врнежите, ET0
    и Kc ненегативни; инаку се крева ValueError.

    Редоследот е важен:
    1. effective=effectiveness·rain_mm се додава на w. Остатокот од дождот
       е ineffective_rain_mm и не влегува во кореновата зона.
    2. overflow го отстранува вишокот над заситеноста. wet е количеството
       по ова прелевање, пред евапотранспирацијата.
    3. Ks се оценува во wet. Побарувачката е Ks·Kc·ET0; eta ја ограничува
       на max(wet−wp,0), за извлекувањето вода да не го намали W под wp.
       Ако wet веќе е под wp, eta е нула и таа вода останува во моделот.
    4. По eta, drainage отстранува drainage_fraction од вишокот над fc.
       Преостанатото result е g(w), физичката предвидена процена во mm.

    Враќа (result, flux). Во flux, rain_mm и effective_rain_mm се влезните
    количества; ineffective_rain_mm, overflow_mm, eta_mm и drainage_mm
    се ненегативни излезни количества за денот, сите во mm. irrigation_mm
    е секогаш 0: се претпоставува земјоделство без наводнување.
    ks е бездимензионален, wet_storage_mm е wet, physical_prediction_mm
    е result. balance_error_mm е остатокот од
    result − (w + rain − ineffective_rain − overflow − eta − drainage),
    кој треба да е нула до нумеричко заокружување. Корекциите на филтерот
    не се физички текови и не се вклучени во оваа проверка.

    Нема капиларен доток, подземна вода, странични текови, посебен модел
    на инфилтрација/површинско истекување, снег или замрзнување. Испарувањето
    од почвата и транспирацијата се здружени во eta, без одделни состојби.
    """
    if not all(math.isfinite(v) for v in (w, rain_mm, et0_mm, kc)):
        raise ValueError('Nonfinite state/forcing')
    if not 0 <= w <= p.saturation or min(rain_mm, et0_mm, kc) < 0:
        raise ValueError('State/forcing outside physical domain')
    # Прво влегува ефективниот дожд; Ks ќе се оцени со оваа дополнителна вода.
    effective = p.effectiveness * rain_mm
    wet = w + effective
    overflow = max(wet - p.saturation, 0.0)
    wet -= overflow
    ks = stress(wet, p)
    eta = min(ks * kc * et0_mm, max(wet - p.wp, 0.0))
    after_et = wet - eta
    # Дренажата го гледа вишокот по евапотранспирацијата, не веднаш по дождот.
    drainage = p.drainage_fraction * max(after_et - p.fc, 0.0)
    result = after_et - drainage
    flux = dict(rain_mm=rain_mm, effective_rain_mm=effective, irrigation_mm=0.0,
                ineffective_rain_mm=rain_mm-effective, overflow_mm=overflow,
                eta_mm=eta, drainage_mm=drainage, ks=ks, wet_storage_mm=wet,
                physical_prediction_mm=result)
    flux['balance_error_mm'] = result - (w + rain_mm - flux['ineffective_rain_mm'] - overflow - eta - drainage)
    return result, flux

def water_balance_jacobian(w, rain_mm, et0_mm, kc, p):
    """Врати бездимензионален G_t=∂g/∂W од активните гранки на budget.

    runner.run го оценува во w=before, старата корегирана процена во mm,
    за EKF и наследениот OpenLoop.predict. rain_mm, et0_mm, kc и Parameters p
    ги имаат истите единици и ограничувања како во budget. При изведувањето
    се држат фиксни времето, дневните врнежи, ET0, тековниот Kc и сите
    параметри. Ова не е извод на Kc по време или извод по метеоролошки влез.

    d_wet е 1 до заситеноста вклучително, а 0 при прелевање: дополнителна
    почетна вода тогаш целосно истекува. d_stress е d_wet/((1−p.p)·TAW)
    само во отворениот интервал 0<stress_argument<1, а инаку е 0.
    d_available е d_wet само над wp. d_eta ја следи побарувачката кога
    demand<available, а инаку ја следи достапната вода. d_after_et ги
    одзема тие загуби; над fc дренажата одзема уште drainage_fraction
    од таа чувствителност. Вратениот извод е d_after_et−d_drainage.

    На недиференцијабилните прагови ова се избрани гранки, не единствен
    класичен извод: при точно заситување d_wet=1; при двата прага на Ks
    d_stress=0; при wet=wp, d_available=0; при demand=available се избира
    d_available; при after_et=fc, d_drainage=0. Особено, на горниот праг
    на Ks се избира заситената гранка Ks=1, од страната со повеќе вода.
    Затоа општо правило „секогаш гранката со помалку вода“ не важи.
    """
    if not all(math.isfinite(v) for v in (w, rain_mm, et0_mm, kc)):
        raise ValueError('Nonfinite state/forcing')
    if not 0 <= w <= p.saturation or min(rain_mm, et0_mm, kc) < 0:
        raise ValueError('State/forcing outside physical domain')

    # Сите d_* подолу се изводи по почетното w, со фиксни дневни влезови.
    wet_before_overflow = w + p.effectiveness * rain_mm
    if wet_before_overflow <= p.saturation:
        wet = wet_before_overflow
        d_wet = 1.0
    else:
        wet = p.saturation
        d_wet = 0.0

    stress_denominator = (1 - p.p) * p.taw
    stress_argument = (wet - p.wp) / stress_denominator
    if stress_argument <= 0:
        d_stress = 0.0
        stress_value = 0.0
    elif stress_argument >= 1:
        # И точно на горниот праг се зема рамната гранка Ks=1, со извод 0.
        d_stress = 0.0
        stress_value = 1.0
    else:
        d_stress = d_wet / stress_denominator
        stress_value = stress_argument

    available = max(wet - p.wp, 0.0)
    d_available = d_wet if wet > p.wp else 0.0
    demand = stress_value * kc * et0_mm
    d_demand = d_stress * kc * et0_mm
    if demand < available:
        d_eta = d_demand
    else:
        # При еднакви вредности на min се избира изводот на достапната вода.
        d_eta = d_available

    d_after_et = d_wet - d_eta
    after_et = wet - min(demand, available)
    if after_et > p.fc:
        d_drainage = p.drainage_fraction * d_after_et
    else:
        d_drainage = 0.0
    return d_after_et - d_drainage

def finite_difference_derivative(f, x, scale=1.0, lower=-math.inf, upper=math.inf):
    """Нумерички извод за пренос на метеоролошката неизвесност во runner.run.

    f враќа W во mm; x е дождот или ET0 во mm, со фиксна почетна состојба.
    scale ја задава големината за чекорот (runner користи TAW во mm),
    а lower/upper се дозволени граници во единиците на x. Централната
    разлика се скратува кај границите; за дожд и ET0 долната е нула.
    Враќа Δf/Δx, тука mm/mm; крева ValueError ако нема интервал.
    Кај праг може да опфати две гранки. Ова не го заменува аналитичкиот G_t
    по состојбата; f се повикува двапати, без менување на филтерот.
    """
    step = np.cbrt(np.finfo(float).eps) * max(abs(x), abs(scale), 1.0)
    lo, hi = max(lower, x-step), min(upper, x+step)
    if hi <= lo: raise ValueError('No finite-difference interval')
    return (f(hi)-f(lo))/(hi-lo)

def calendar_kc(day, planting_date, crop):
    """Врати (Kc, фаза) од календарот што го користи runner.input_history.

    day и planting_date се датуми YYYY-MM-DD; crop е редот од crops.csv
    со *_days и kc_initial/kc_mid/kc_end. Kc е бездимензионален: константен
    во initial и mid, линеарно менлив во development и late.
    Денот на садење има возраст 0; ден пред садење или на/по крајот на
    збирот на фазите крева ValueError. Овој календар се проверува и кога
    подоцна Kc се избира од NDVI. Нема промена на crop.
    """
    age = (date.fromisoformat(str(day)) - date.fromisoformat(planting_date)).days
    a,b,c,d = [int(crop[k+'_days']) for k in ('initial','development','mid','late')]
    ki,km,ke = [float(crop['kc_'+k]) for k in ('initial','mid','end')]
    if age < 0 or age >= a+b+c+d: raise ValueError('Date outside assumed crop calendar')
    if age < a: return ki, 'initial'
    if age < a+b: return ki+(km-ki)*(age-a)/b, 'development'
    if age < a+b+c: return km, 'mid'
    return km+(ke-km)*(age-a-b-c)/d, 'late'

def ndvi_kc(ndvi, crop):
    """Претвори претходно прифатен NDVI во бездимензионален Kc.

    runner.input_history го користи само во режим ndvi, со доволно свеж
    индекс од претходен ден. ndvi е бездимензионален во [-1,1], проверен
    од повикувачот; crop ги дава kc_* и default_ndvi од референтната табела.
    Врската е линеарна пред ограничувањето меѓу min(ki,ke) и max(ki,km,ke).
    За wheat, ki=0.40 е фиксна вредност во кодот, а не поставка од табелата.
    Ова го задава влезот Kc; не создава набљудување на W или корекција.
    """
    ki,km,ke = [float(crop['kc_'+k]) for k in ('initial','mid','end')]
    # Во опционалниот NDVI режим, за wheat се користи фиксното ki=0.40.
    if crop['crop'] == 'wheat': ki = 0.40
    slope = (km-ki)/(max(float(crop['default_ndvi']), .25)-.20)
    return float(np.clip(ki+slope*(ndvi-.20), min(ki,ke), max(ki,km,ke)))
