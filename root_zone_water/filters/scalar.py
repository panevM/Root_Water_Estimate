"""Скаларни EKF, UKF и филтер само со предвидување за водниот биланс.

Состојбата е x_t = W_t, количество вода во кореновата зона во mm.
runner.run избира класа преку FILTERS и го задржува истиот објект секој ден.
Пред predict, mean и variance се μ_{t-1} и Σ_{t-1}; потоа се предвидува
μ̄_t и нејзината коваријанса Σ̄_t. По update се корегираната процена
μ_t и Σ_t. variance е скаларна коваријанса во mm², не стандардна девијација.

transition е g со веќе зададени дневни влезови. process_covariance е R_t,
коваријанса на шумот на процесот; measurement_covariance е Q_t, коваријанса
на шумот на мерењето. Заедничкиот update користи h(W_t) = W_t и H_t = 1:
набљудувањето веќе е индиректна процена на количеството вода добиена од NDMI
во mm, а не суров NDMI или директно теренско мерење.
"""
from abc import ABC, abstractmethod
import math
import numpy as np

class ScalarFilter(ABC):
    """Заедничка состојба и корекција наследени од конкретните филтри.

    predict, update и constrain ги менуваат mean и variance на истиот објект.
    result и проверките само читаат; тука нема датотеки или мрежни повици.
    """
    def __init__(self, mean, variance, bounds=None):
        """Зачувај почетна процена mean (mm), variance (mm²) и граници.

        runner.run задава bounds=(0, p.saturation); None значи без проекција.
        Границите се пар (долна, горна) во mm. Се бара конечна средна вредност
        и конечна ненегативна коваријанса. Иницијализацијата ги проверува,
        но не ја ограничува средната вредност на bounds.
        """
        self.mean, self.variance, self.bounds = float(mean), float(variance), bounds
        self.check()

    def check(self):
        """Крени ValueError за неконечна процена или негативна коваријанса.

        Се повикува при создавање и по constrain; не проверува bounds
        и не ја менува состојбата.
        """
        if not math.isfinite(self.mean) or not math.isfinite(self.variance) or self.variance < 0:
            raise ValueError('Invalid state or scalar covariance')

    def constrain(self):
        """Проектирај ја mean во bounds и врати го поместувањето во mm.

        predict и update го повикуваат овој чекор по своите равенки.
        При проекција variance се зголемува за квадратот на поместувањето
        во mm². Ова е дополнителна хеуристика за неизвесноста, а не стандардна
        Калманова равенка или точна коваријанса на отсечена распределба.
        Не гарантира калибрирана неизвесност и не ги ограничува нејзините
        интервали. Без bounds само ја проверува состојбата и враќа нула.
        """
        before = self.mean
        if self.bounds is not None:
            self.mean = float(np.clip(self.mean, *self.bounds))
            # Ја задржуваме коваријансата и додаваме квадрат на поместувањето.
            self.variance += (self.mean-before)**2
        self.check()
        return self.mean-before

    @abstractmethod
    def predict(self, transition, process_covariance, scale=1.0):
        """Договор за дневно предвидување што ги менува mean и variance.

        transition пресликува W во mm во следно W во mm; process_covariance
        е R_t во mm² и мора да биде конечна и ненегативна. scale е задржан
        аргумент на интерфејсот (runner праќа TAW во mm), но конкретните
        филтри тука не го користат. Имплементациите враќаат дијагностички
        речник; EKF дополнително бара однапред пресметан state_jacobian.
        """
        pass

    def update(self, observation, measurement_covariance):
        """Корегирај ја предвидената процена со едно прифатено набљудување.

        runner.run го повикува наследениот метод за EKF и UKF по predict.
        observation е z_t, индиректна процена на количеството вода добиена
        од NDMI во mm. measurement_covariance е Q_t во mm², строго позитивна;
        двата аргумента мора да се конечни, инаку се крева ValueError.

        Бидејќи h(W)=W и H_t=1, innovation е ν_t=z_t−μ̄_t (mm),
        innovation_covariance е S_t=Σ̄_t+Q_t (mm²), а gain е бездимензионалното
        Калманово засилување K_t=Σ̄_t/S_t. Потоа mean станува
        μ_t=μ̄_t+K_t ν_t, а variance станува Σ_t=(1−K_t)Σ̄_t.
        Ова е скаларната коваријансна корекција од курсот; constrain потоа
        може дополнително да ги промени и процената и коваријансата.

        Враќа innovation_mm, S_mm2, gain и posterior_bound_adjustment_mm
        (поместувањето од последната проекција, во mm). На ден без прифатен
        NDMI runner не го повикува методот: предвидената процена се пренесува
        понатаму. OpenLoop го наследува методот, но runner го прескокнува.
        """
        if (not math.isfinite(observation) or not math.isfinite(measurement_covariance)
                or measurement_covariance <= 0):
            raise ValueError('Observation and measurement covariance must be finite; covariance positive')
        # Иновација: ν_t = z_t − h(μ̄_t), со h(x)=x и H_t=1.
        innovation = observation-self.mean
        # Коваријанса на иновацијата: S_t = H_t Σ̄_t H_t^T + Q_t.
        innovation_covariance = self.variance+measurement_covariance
        # Калманово засилување: K_t = Σ̄_t H_t^T S_t^-1.
        gain = self.variance/innovation_covariance
        # Корегирана процена: μ_t = μ̄_t + K_t ν_t.
        self.mean += gain*innovation
        # Корегирана коваријанса: Σ_t = (I−K_t H_t)Σ̄_t.
        self.variance = (1-gain)*self.variance
        bound_adjustment = self.constrain()
        return dict(innovation_mm=innovation, S_mm2=innovation_covariance, gain=gain,
                    posterior_bound_adjustment_mm=bound_adjustment)

    def result(self):
        """Врати (mean во mm, variance во mm²) без промена на објектот.

        runner.run чита пред и по предвидувањето и по можната корекција;
        значењето е предвидена или корегирана процена според последниот чекор.
        """
        return self.mean, self.variance

    @staticmethod
    def check_process_covariance(process_covariance):
        """Провери R_t во mm² пред predict; крени ValueError за невалидна вредност.

        Нула е дозволена, негативна или неконечна вредност не е. Методот
        не ја менува процената.
        """
        if not math.isfinite(process_covariance) or process_covariance < 0:
            raise ValueError('Process covariance must be finite and nonnegative')

class EKF(ScalarFilter):
    """EKF со нелинеарна средна прогноза и аналитички Јакобијан за коваријансата."""
    def predict(self, transition, process_covariance, scale=1.0, state_jacobian=None):
        """Пресметај μ̄_t=g(μ_{t-1}) и Σ̄_t=G_t Σ_{t-1} G_t^T+R_t.

        runner.run го врзува transition со дневните врнежи, ET0, Kc и
        Parameters. state_jacobian е бројот G_t=∂g/∂W, пресметан таму со
        water_balance_jacobian во старата корегирана процена before,
        со истите дневни влезови. Не е функција и не се оценува во μ̄_t.
        Производот jacobian*jacobian*self.variance е скаларниот G_t Σ G_t^T.

        process_covariance е R_t во mm²; scale (mm) не се користи.
        Ги менува mean и variance, па повикува constrain. Враќа jacobian
        (бездимензионален), sigma_points_clipped=0 и
        prior_bound_adjustment_mm (поместување од проекцијата во mm).
        Без state_jacobian се крева ValueError, но mean веќе е променета;
        овој метод не ги поништува делумните промени при неуспех.
        """
        self.check_process_covariance(process_covariance)
        # Средната вредност поминува низ самиот g, без линеаризација на g.
        previous_mean = self.mean
        self.mean = transition(previous_mean)
        # Линеаризацијата преку G_t се користи за предвидување на коваријансата.
        jacobian = state_jacobian
        if jacobian is None:
            raise ValueError('EKF requires the analytical state_jacobian')
        self.variance = jacobian*jacobian*self.variance+process_covariance
        adjustment = self.constrain()
        return dict(jacobian=jacobian, sigma_points_clipped=0, prior_bound_adjustment_mm=adjustment)

class UKF(ScalarFilter):
    """UKF што ја пренесува неизвесноста низ g со три сигма-точки."""
    def predict(self, transition, process_covariance, scale=1.0):
        """Предвиди ја распределбата преку g, без извод по состојбата.

        runner.run ги задава transition: mm→mm и process_covariance=R_t
        во mm², како кај EKF; scale е задржан, но неискористен аргумент.
        За n=1, α=1, β=2, κ=0 е λ=α²(n+κ)−n=0. spread=√Σ_{t-1},
        па points се [μ_{t-1}, μ_{t-1}+spread, μ_{t-1}−spread].
        Тежините за средната вредност се [0, 1/2, 1/2], а за коваријансата
        [2, 1/2, 1/2], бидејќи централната добива 1−α²+β=2.

        Пред g, точките се ограничуваат на bounds за budget да добие
        физички дозволени влезови. Ова отсекување е хеуристика: тежините
        не се менуваат и точките веќе не мора да ги претставуваат почетните
        моменти. values се g за секоја ограничена точка; нивната пондерирана
        средина е μ̄_t, а Σ̄_t=Σ_i W_i^c (values_i−μ̄_t)²+R_t.
        Централната точка има нулта тежина за средината, но учествува во
        коваријансата. По овие равенки constrain додава можни гранични промени.

        Ги менува mean и variance. Враќа jacobian=NaN (не се користи),
        sigma_points_clipped (број на отсечени точки) и
        prior_bound_adjustment_mm (поместување на средината во mm).
        За тековното линеарно h(W)=W, S_t=Σ̄_t+Q_t и вкрстената
        коваријанса е Σ̄_t, па наследениот ScalarFilter.update ја дава
        скаларната UKF корекција без нови сигма-точки за набљудувањето.
        """
        self.check_process_covariance(process_covariance)
        # n=1, alpha=1, beta=2, kappa=0: lambda=0; надворешни тежини 1/2.
        spread = math.sqrt(self.variance)
        points = np.array([self.mean, self.mean+spread, self.mean-spread])
        clipped = np.clip(points, *self.bounds) if self.bounds is not None else points
        count = int(np.count_nonzero(points != clipped))
        values = np.array([transition(float(x)) for x in clipped])
        self.mean = float((values[1]+values[2])/2)
        delta = values-self.mean
        self.variance = float(2*delta[0]**2 + .5*delta[1]**2 + .5*delta[2]**2 + process_covariance)
        adjustment = self.constrain()
        return dict(jacobian=math.nan, sigma_points_clipped=count, prior_bound_adjustment_mm=adjustment)

class OpenLoop(EKF):
    """Ги наследува EKF предвидувањето, иницијализацијата и заедничките методи.

    runner.run за 'open_loop' никогаш не повикува update, па секој ден има
    само предвидување. Сепак се пресметуваат средната вредност и коваријансата и, ако има NDMI,
    иновација за споредба. Самата класа не забранува директен повик на update.
    """
