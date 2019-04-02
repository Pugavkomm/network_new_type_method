#include <mass_th.h>
#include <iostream>
using namespace std;
void special_omega_rand(mass_th &omega, double p, double G);
void special_omega_mind(mass_th &omega);
void teaher_f(double *x, double dt, int nt);
void model_neuron(mass_th &v, mass_th &I, mass_th &synaptic, int N, double a, double d, double eps,
				  double beta, double J);
void model_synaps(mass_th &h, mass_th &r, mass_th &hr, double &dt,
				  mass_th &v, double &vpeak, double &vreset, mass_th &ISPC, mass_th &JD, int count, int N, double M, double M1);
int special_omega_find_index(mass_th &omega, mass_th &v, double &vpeak,
							 mass_th &JD);
int main(int argc, char const *argv[])
{
	cout << "start\n";
	srand(time(0));
	omp_set_num_threads(6);
	//omp_set_dynamic(0);
	cout << "Start\n";
	int N = 2000; //Количество нейронов
	double dt = 0.001;
	double T = 100.;		// Общее время интегрирования
	int nt = round(T / dt); // количество итераций
	int i_last = 0;
	double tmin = 20;  // начало обучения
	double tcrit = 60; // конец обучения
	int imin = round(tmin / dt);
	int icrit = round(tcrit / dt);
	int count;	// количество элементов ???
	int step = 0; // шаг работы RLS - method
	//было для двуруслового синуса
	//double G = 0.01; // Параметр стат для стат весов
	//double Q = .11;	 // Параметр для обучаеемых весов
	//для хаоса
	double Q = .1;
	double G = 0.01;
	double lambda = .01; //Скорость обучения
	double M = 1.01;	 // Параметр хар. затухание в синапсах
	double M1 = 1.95;	// Параметр хар. затухание в синапсах
	double p = .1;
	double divive;
	double const_p = .4;
	//____________________________________________________________
	//Параметры модели
	double a, beta, d, eps, J;
	//eps = 0.00793; beta = 0.19547; d = .5012; a = 0.25;
	eps = 0.005;
	beta = 0.018;
	d = .26;
	a = 0.25;
	J = 0.15;
	double vpeak = .2;
	double vreset = (1 + a - sqrt(1.0 - a + a * a)) / 3;
	double td = 0.1;
	double tr = 1;

	double a1, b1, c1;

	a1 = 1;
	b1 = -(2 - 1 / M - 1 / M1);
	c1 = (1 - 1 / M) * (1 - 1 / M1);

	double s1 = -b1 + sqrt(b1 * b1 - 4 * a1 * c1);
	s1 /= 2 * a1;
	double s2 = -b1 - sqrt(b1 * b1 - 4 * a1 * c1);
	s1 /= 2 * a1;

	cout << "s1 = " << s1 << "\ns2 = " << s2;
	//cin >> s1;

	//____________________________________________________________
	cout << "G = " << G
		 << ",Q = " << Q
		 << ", M = " << M
		 << ", lambda = " << lambda
		 << '\n';
	//######################################################
	// OMEGA - матрица весов
	mass_th OMEGA(N, N);
	mass_th save(N);	//для временных нужд
	mass_th index(N);   //для определения индексов (спец функция)
	mass_th E(N);		// Энкодер
	mass_th BPhi(N);	// декодер
	mass_th IPSC(N);	// часть тока
	mass_th JD(N);		// вторая часть тока
	mass_th JX(N);		// полный так
	mass_th cd(N);		// учавствует в минимищации ошибки
	mass_th Pinv(N, N); // RLS - method
	//#####################################################
	int step_save_spatial = 4000;
	mass_th
		mass_pre_learning(step_save_spatial, N);
	mass_pre_learning.zero();
	mass_th mass_learning(step_save_spatial, N);
	mass_learning.zero();
	mass_th mass_post_learning(step_save_spatial, N);
	mass_post_learning.zero();
	//#####################################################
	//
	//Формируем массивы для выражений синапсов
	mass_th h(N);
	mass_th hr(N);
	mass_th r(N);
	// разбрасываем OMEGA и делаем среднее нулевым
	special_omega_rand(OMEGA, p, G);
	special_omega_mind(OMEGA);
	// Разбрасываем E
	E.random(-1, 1);
	E = E * Q;
	//Формируем массивы для выражений синапсов
	//
	//Обнуляем основные массивы (на всякий)
	save.zero();
	BPhi.zero();
	IPSC.zero();
	h.zero();
	r.zero();
	hr.zero();
	cd.zero();
	JD.zero();
	//_-_-_---___--_--__-_
	Pinv.eye();			  // единичная матрица
	Pinv = Pinv * lambda; // домножили на lambda
	//_-_-_---___--_--__-_
	//_-_-_---___--_--__-_
	// зададим переменные модели (напряжение и ток)
	mass_th I(N);
	mass_th v(N);
	// разбрасываем напряжение в случайном порядке от vreset до vpeak
	v.random(vreset, .2);
	// обнуляем ток
	I.zero();

	double z = 0;				 // выход сети
	double err = 0;				 // ошибка
	double *xz = new double[nt]; // teacher

	teaher_f(xz, dt, nt);
	double start_time = omp_get_wtime();

	cout << "Параметры системы\n"
		 << "G = " << G << ", Q = " << Q << "\n";

	///////////////////////START ALGORITM//////////////////////////
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	for (int step_system; step_system < nt; step_system++)
	{
		JX = E * z;

		JX = JX + IPSC;
		model_neuron(v, I, JX, N, a, d, eps, beta, J);
		count = special_omega_find_index(OMEGA, v, vpeak, JD);
		model_synaps(h, r, hr, dt, v, vpeak, vreset, IPSC, JD, count, N, M, M1);
		//Fauto zs = ((BPhi--) * r);
		z = ((BPhi--) * r).matrix[0][0];
		cout << z;
		cout << '\n';
	}

	///////////////////////START ALGORITM//////////////////////////
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	cout << "Test program start cmake" << endl;
	return 0;
}
void special_omega_rand(mass_th &omega, double p, double G)
{
	default_random_engine generator;
	normal_distribution<double> distribution(0.0, 1.0);
	for (int i = 0; i < omega.row; i++)
	{
		for (int j = 0; j < omega.col; j++)
		{
			omega.matrix[i][j] = G * distribution(generator) * ((double)rand() / RAND_MAX < p);

			omega.matrix[i][j] /= (sqrt((double)omega.row) * p);
		}
	}
}

void special_omega_mind(mass_th &omega)
{
	// спец функция для создания разреженной матрицы по опр закону
	// который задали авторы статьи
	int count;
	double mine_swipe;
	int *save = new int[omega.row];
	for (int i = 0; i < omega.row; i++)
	{
		count = 0;
		for (int j = 0; j < omega.col; j++)
		{
			if (abs(omega.matrix[i][j]) != 0.0)
			{
				save[count] = j;
				count++;
			}
		}
		mine_swipe = 0.0;
		for (int j = 0; j < count; j++)
			mine_swipe += omega.matrix[i][save[j]];
		mine_swipe /= count;
		for (int j = 0; j < count; j++)
			omega.matrix[i][save[j]] -= mine_swipe;
	}
	delete[] save;
}
void teaher_f(double *x, double dt, int nt)
{
	for (int i = 0; i < nt; i++)
	{
		x[i] = sin(2 * M_PI * i * dt * 5);
	}
}

void model_neuron(mass_th &v, mass_th &I, mass_th &synaptic, int N, double a, double d, double eps,
				  double beta, double J)
{
	int i;
	double I_save;
#pragma omp parallel shared(v, I, beta, eps, J, a, d, synaptic, N) private(i, I_save)
	{
#pragma omp for schedule(dynamic)
		for (i = 0; i < N; i++)
		{
			I_save = I.matrix[i][0];
			I.matrix[i][0] = I.matrix[i][0] + eps * (v.matrix[i][0] - J);
			v.matrix[i][0] = v.matrix[i][0] + v.matrix[i][0] * (v.matrix[i][0] - a) * (1 - v.matrix[i][0]) -
							 beta * (v.matrix[i][0] > d) - I_save + synaptic.matrix[i][0];
		}
	}
}

void model_synaps(mass_th &h, mass_th &r, mass_th &hr, double &dt,
				  mass_th &v, double &vpeak, double &vreset, mass_th &ISPC, mass_th &JD, int count, int N, double M, double M1)
{
#pragma omp parallel shared(h, r, hr, dt, v, vpeak, vreset, ISPC, JD, count, N, M, M1)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < N; i++)
		{
			ISPC.matrix[i][0] += -ISPC.matrix[i][0] / M + h.matrix[i][0];
			h.matrix[i][0] += -h.matrix[i][0] / M1 + JD.matrix[i][0] * (count > 0) / (M * M1);
			r.matrix[i][0] += -r.matrix[i][0] / M + hr.matrix[i][0];
			hr.matrix[i][0] += -hr.matrix[i][0] / M1 + (v.matrix[i][0] > vpeak) / (M * M1);
		}
	}
}
int special_omega_find_index(mass_th &omega, mass_th &v, double &vpeak,
							 mass_th &JD)
{
	int i, j;
	int count = 0;
	int *save = new int[omega.row];

	for (i = 0; i < omega.row; i++)
	{
		if (v.matrix[i][0] >= vpeak)
		{
			save[count] = i;
			count++;
		}
	}
	if (count > 0)
		for (i = 0; i < omega.row; i++)
		{
			JD.matrix[i][0] = 0;
			for (j = 0; j < count; j++)
			{
				JD.matrix[i][0] += omega.matrix[i][save[j]];
			}
		}
	delete[] save;
	return count;
}
