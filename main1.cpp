#include <iostream>
#include <omp.h>
#include "mass_th.h"
using namespace std;

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

  void teaher_f(double *, double , int);

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
									mass_th &v, double &vpeak, double &vreset, mass_th &ISPC, mass_th &JD, int count, int N, double M)
{
#pragma omp parallel shared(h, r, hr, dt, v, vpeak, vreset, ISPC, JD, count, N, M)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < N; i++)
		{
			ISPC.matrix[i][0] = ISPC.matrix[i][0] - ISPC.matrix[i][0] / M + 1 / M * JD.matrix[i][0] * (count > 0);
			r.matrix[i][0] = r.matrix[i][0] - r.matrix[i][0] / M + 1 / M * (v.matrix[i][0] >= vpeak);
			//r.matrix[i][0] = r.matrix[i][0] - r.matrix[i][0]/M +  (v.matrix[i][0]);
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

void new_model_synaps(mass_th &r, mass_th &v, double vpeak, double const_p, int N)
{
#pragma omp parallel shared(r, v, vpeak, const_p, N)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < N; i++)
		{
			r.matrix[i][0] = const_p * v.matrix[i][0] * (v.matrix[i][0] > vpeak);
		}
	}
}

void copy_spatial(mass_th &a, mass_th &matrix, int &step_system, int &col)
{
#pragma omp parallel shared(a, matrix, step_system, col)
	{
#pragma omp for schedule(dynamic)
		for (int j = 0; j < col; j++)
			a.matrix[step_system][j] = matrix.matrix[j][0];
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

	void rls(mass_th &BPhi, mass_th &Pinv, mass_th &cd,
				 double &err, double divive, int N)
{
	int i, j;
#pragma omp parallel shared(BPhi, Pinv, cd, err, divive, N) private(i, j)
	{
#pragma omp for schedule(dynamic)
		for (i = 0; i < N; i++)
		{
			BPhi.matrix[i][0] -= (cd.matrix[i][0] * err);
			for (j = 0; j < N; j++)
				Pinv.matrix[i][j] -= (cd.matrix[i][0] * cd.matrix[j][0]) / (1.0 + divive);
		}
	}
}
int main(int argc, char const *argv[]) {
  cout << "start\n";
	srand(time(0));
	omp_set_num_threads(6);
	//omp_set_dynamic(0);
	cout << "Start\n";
	int N = 2000; //Количество нейронов
	double dt = 0.001;
	double T = 100.;				// Общее время интегрирования
	int nt = round(T / dt); // количество итераций
	int i_last = 0;
	double tmin = 20; // начало обучения
	double tcrit = 60; // конец обучения
	int imin = round(tmin / dt);
	int icrit = round(tcrit / dt);
	int count;		// количество элементов ???
	int step = 0; // шаг работы RLS - method
	//было для двуруслового синуса
	//double G = 0.01; // Параметр стат для стат весов
	//double Q = .11;	 // Параметр для обучаеемых весов
	//для хаоса
	double Q =  .1;
	double G = 0.01;
	double lambda = .01; //Скорость обучения
	double M = 1.01;		 // Параметр хар. затухание в синапсах
	double M1 = 1.95;		 // Параметр хар. затухание в синапсах
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
	//eps = 0.005; beta = 0.04; d = .5; a = 0.25;
	//eps = 0.1; beta = 0.04; d = .5; a = 0.25;
	//J = (1.0 + a - sqrt(1.0 - a + a * a + 3 * eps)) / 3 + 0.05;
	J = 0.15;
	//J = .15; eps = .005; beta = .04; d = .5; a = .25; // периодические
	//J = .1; eps = .001; beta = .3; d = .45; a = .1;
	//double vpeak = (1 + a + sqrt(1.0 - a + a * a)) / 3;
	//double vpeak = .2;
//	a = .2; J = .14; eps = .02; beta = .265; d = .45; // хаос dfr ~ 1.3
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
	cin >> s1;

	//____________________________________________________________
	cout << "G = " << G
			 << ",Q = " << Q
			 << ", M = " << M
			 << ", lambda = " << lambda
			 << '\n';
	//######################################################
	// OMEGA - матрица весов
  mass_th OMEGA(N, N);
	mass_th save(N);		//для временных нужд
	mass_th index(N);		//для определения индексов (спец функция)
	mass_th E(N);				// Энкодер
	mass_th BPhi(N);		// декодер
	mass_th IPSC(N);		// часть тока
	mass_th JD(N);			// вторая часть тока
	mass_th JX(N);			// полный так
	mass_th cd(N);			// учавствует в минимищации ошибки
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
	Pinv.eye();									 // единичная матрица
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

	double z = 0;								 // выход сети
	double err = 0;							 // ошибка
	double *xz = new double[nt]; // teacher

  teaher_f(xz, dt, nt);

  cout << "Параметры системы\n"
			 << "G = " << G << ", Q = " << Q << "\n";

	///////////////////////START ALGORITM//////////////////////////
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	//включение и отключение сохранения в массивы
	bool record_save_spatial = true;
	bool record_save_omega = true;
	bool record_save_neurons = true;
	bool record_save_synaptic = true;
	//
	// Всякие дополнительные бессполезные вещи, которые нужны
	// для теста и настройки сети)
	double save1 = 0;
	double save2 = 0;
	double save3 = 0;
	double save4 = 0;
	double save5 = 0;
	// для 5 нейронов
	double *savem1 = new double[nt]; // 1 нейрон
	double *savem2 = new double[nt]; // 2 нейрон
	double *savem3 = new double[nt]; // 3 нейрон
	double *savem4 = new double[nt]; // 4 нейрон
	double *savem5 = new double[nt]; // 5 нейрон
	// для 5 нейронов
	// r
	double *savem6 = new double[nt];
	double *savem7 = new double[nt];
	double *savem8 = new double[nt];
	double *savem9 = new double[nt];
	double *savem10 = new double[nt];
	//hr
	double *savem11 = new double[nt];
	double *savem12 = new double[nt];
	double *savem13 = new double[nt];
	double *savem14 = new double[nt];
	double *savem15 = new double[nt];
	//для сохранения phi
	double *decoder1 = new double[nt];
	double *decoder2 = new double[nt];
	double *decoder3 = new double[nt];
	double *decoder4 = new double[nt];
	double *decoder5 = new double[nt];
	// синаптические токи
	double *savemz = new double[nt]; // выход

	double *save_err = new double[nt]; // сохранение ошибки
	// Создадим потоки для работы с файлами
	ofstream file1;				 // сохранение нейронов
	ofstream file2;				 // информация о параметрах
	ofstream file3;				 // r
	ofstream file4;				 // hr
	ofstream file5;				 // ошибка
	ofstream file6;				 // выходы
	ofstream file7;				 //пространственно временной хаос
	ofstream file_decoder; //декодеры
	ofstream file_omega;	 //Омега
	// И сами файлы
	//file1
	file1.open("file1.txt");
	file1.close();
	file1.open("file1.txt", ios::app);
	//file2
	file2.open("file2.txt");
	file2.close();
	file2.open("file2.txt", ios::app);
	//file3
	file3.open("file3.txt");
	file3.close();
	file3.open("file3.txt", ios::app);
	//file4
	file4.open("file4.txt");
	file4.close();
	file4.open("file4.txt", ios::app);
	//file5
	file5.open("file5.txt");
	file5.close();
	file5.open("file5.txt", ios::app);
	//file6
	file6.open("file6.txt");
	file6.close();
	file6.open("file6.txt", ios::app);
	//file7
	file7.open("file7.txt");
	file7.close();
	file7.open("file7.txt", ios::app);
	//file_decoder
	file_decoder.open("file_decoder.txt");
	file_decoder.close();
	file_decoder.open("file_decoder.txt", ios::app);
	//file_omega
	file_omega.open("file_omega.txt");
	file_omega.close();
	file_omega.open("file_omega.txt", ios::app);
	double start_time = omp_get_wtime();
	int step_for_save = 0;
	int indicator1 = 1;
	int indicator2 = 1;
	int indicator3 = 1;
	//___________________________________________________________
for (int step_system; step_system < nt; step_system++)
{
  JX = E  * z;

  JX = JX + IPSC;
  model_neuron(v, I, JX, N, a, d, eps, beta, J);
  if (record_save_spatial)
		{
			if (step_system < step_save_spatial)
			{
				copy_spatial(mass_pre_learning, v, step_for_save, N);
				step_for_save++;
			}
			else if (step_system >= imin && step_system < imin + step_save_spatial)
			{
				if (indicator2 == 1)
				{
					indicator2 = 0;
					step_for_save = 0;
				}
				copy_spatial(mass_learning, v, step_for_save, N);
				step_for_save++;
			}
			else if (step_system >= icrit && step_system < icrit + step_save_spatial)
			{
				if (indicator3 == 1)
				{
					indicator3 = 0;
					step_for_save = 0;
				}
				copy_spatial(mass_post_learning, v, step_for_save, N);
				step_for_save++;
			}
		}
    //сохраняем 5 случайно выбранных нейронов
		if (record_save_neurons)
		{
			savem1[step_system] = v.matrix[0][0];
			savem2[step_system] = v.matrix[100][0];
			savem3[step_system] = v.matrix[520][0];
			savem4[step_system] = v.matrix[800][0];
			savem5[step_system] = v.matrix[900][0]; 
		}
    count = special_omega_find_index(OMEGA, v, vpeak, JD);
    model_synaps(h, r, hr, dt, v, vpeak, vreset, IPSC, JD, count, N, M, M1);
    //Fauto zs = ((BPhi--) * r);
    z = ((BPhi--) * r).matrix[0][0];



    savemz[step_system] = z;
		err = z - xz[step_system];
    //cout << xz[step_system] << '\n';
		save_err[step_system] = err;

    if (step_system > imin)
			if (step_system < icrit)
      {
				//auto rT = r--;
				//auto cdT = cd--;
        cd = Pinv * r;
       //BPhi = BPhi - (cd * err);
        divive = (((r--) * cd).matrix[0][0]);
			rls(BPhi, Pinv, cd, err, divive, N);
			//auto tip = ((r--) * cd);
				//Pinv = Pinv - (cd * cd--) / (1.0 + divive);
      }

      if (record_save_synaptic)
		{
		//Сохраняем в массивы r
			savem6[step_system] = r.matrix[0][0];
			savem7[step_system] = r.matrix[100][0];
			savem8[step_system] = r.matrix[520][0];
			savem9[step_system] = r.matrix[800][0];
			savem10[step_system] = r.matrix[900][0];
			//Сохраняем в массивы h
			savem11[step_system] = hr.matrix[0][0];
			savem12[step_system] = hr.matrix[100][0];
			savem13[step_system] = hr.matrix[520][0];
			savem14[step_system] = hr.matrix[800][0];
			savem15[step_system] = hr.matrix[900][0];
			//Сохраняем в массивы Bphi
			decoder1[step_system] = BPhi.matrix[0][0];
			decoder2[step_system] = BPhi.matrix[100][0];
			decoder3[step_system] = BPhi.matrix[520][0];
			decoder4[step_system] = BPhi.matrix[800][0];
			decoder5[step_system] = BPhi.matrix[900][0];
		}
		save1 += JX.matrix[0][0];

		if (step_system % 100 == 1)
		{

			cout
					<< "err = "
					<< err
					<< ", z = " << z
					<< ", step = "
					<< step_system
					<< ", x1 = " << v.matrix[0][0];
			cout
					<< ", time = " << omp_get_wtime() - start_time
					<< '\n';

			start_time = omp_get_wtime();
			cout << (save1 / (step_system + 1)) << '\n';
			//<< "v1 = " << v.b[0]
			//<< ", v100 = " << v.b[99]
			//<< ", v300 = " << v.b[299]
			//<<'\n';
		}
	}
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	///////////////////////END ALGORITM//////////////////////////
	cout << "END\n";
	cout << "start write file\n";
	file2
			<< "G = " << G << ", Q = " << Q
			<< ", a = " << a << ", J = " << J
			<< ", $beta$ = " << beta
			<< ", $eps = $" << eps
			<< ", d = " << d
			<< ", vpeak = " << vpeak
			<< ", step_system = " << nt
			<< ",start_learn = " << imin
			<< ",  end_learn = " << icrit;
	for (int i = 0; i < nt; i++)
	{
		file1
				<< savem1[i] << ' '
				<< savem2[i] << ' '
				<< savem3[i] << ' '
				<< savem4[i] << ' '
				<< savem5[i] << ' '
				<< vpeak << '\n';
		////////////////////////////////////////
		//save r
		file3
				<< savem6[i] << ' '
				<< savem7[i] << ' '
				<< savem8[i] << ' '
				<< savem9[i] << ' '
				<< savem10[i] << ' '
				<< '\n';
		//save h
		file4
				<< savem11[i] << ' '
				<< savem12[i] << ' '
				<< savem13[i] << ' '
				<< savem14[i] << ' '
				<< savem15[i] << ' '
				<< '\n';
		////////////////////////////////////////
		file5
				<< save_err[i]
				<< '\n';
		////////////////////////////////////////
		file6
				<< xz[i] << ' '
				<< savemz[i]
				<< '\n';

		file_decoder
				<< decoder1[i] << ' '
				<< decoder2[i] << ' '
				<< decoder3[i] << ' '
				<< decoder4[i] << ' '
				<< decoder5[i] << ' '
				<< '\n';
	}

	for (int i = 0; i < 3; i++)
		for (int k = 0; k < step_save_spatial; k++)
		{
			for (int j = 0; j < N; j++)
				switch (i)
				{
				case 0:
					file7 << mass_pre_learning.matrix[k][j] << ' ';

					break;
				case 1:
					file7 << mass_learning.matrix[k][j] << ' ';

					break;
				case 2:
					file7 << mass_post_learning.matrix[k][j] << ' ';
				default:
					break;
				}
			file7 << '\n';
		}

	for (int k = 0; k < 2; k++)
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				switch (k)
				{
				case 0:
					file_omega << OMEGA.matrix[i][j] << ' ';
					break;
				case 1:
					file_omega << (E.matrix[i][0] * BPhi.matrix[j][0]) << ' ';
				default:
					break;
				}
			file_omega << '\n';
		}
	file1.close();
	file2.close();
	file3.close();
	file4.close();
	file5.close();
	file6.close();
	file7.close();
	file_decoder.close();
	file_omega.close();

	cout << "end write file\n";


  return 0;
}


void teaher_f(double *x, double dt, int nt)
{
	for (int i = 0; i < nt; i++)
	{
		x[i] = sin(2 * M_PI * i * dt * 5);
	}
}
