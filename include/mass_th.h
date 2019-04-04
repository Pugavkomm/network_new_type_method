
#ifndef MASS_TH
#define MASS_TH
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <utility>
#include <iomanip> //Манипуляция с выводом
#include <fstream> //Сохранение в файл
#include <math.h>	//Мат функции
#include <ctime>	 //Понадобиться для рандома
#include <random>
#include <omp.h>
#include <math.h>

using namespace std;

class mass_th
{
public:
  int row;
	string name;
  int col;
  double **matrix;


  mass_th(int r, int c);
  mass_th(int r);
	mass_th(int r, int c, string names);
	mass_th(int r, string names);
  mass_th(const mass_th &other) : mass_th(other.row, other.col, other.name)
{
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
        matrix[i][j] = other.matrix[i][j];
}


~mass_th()
{
		//cout << matrix << "\n";
  for (int i = 0; i < row; i++)
    delete[] matrix[i];
  delete[] matrix;
//cout <<this << "- Destruct\n";
}
mass_th &operator=(mass_th other) noexcept
{

    if (this == &other)
      return *this;
    std::swap(row, other.row);
    std::swap(matrix, other.matrix);
		std::swap(col, other.col);
		std::swap(name, other.name);
    return *this;
	//	mass_th out(row, col);
		//for (int i = 0; i < row; i++)
		//	for (int j = 0; j < col; j++)
			//	out.matrix[i][j] = matrix[i][j];
		//return out;
}
mass_th(mass_th &&other) noexcept:
row(exchange(other.row, 0)),
col(exchange(other.col, 0)),
name(exchange(other.name, 0)),
matrix(exchange(other.matrix, nullptr)){}


mass_th operator+(const mass_th& rigth_part) const
{
	if (row != rigth_part.row && col != rigth_part.col)
	{
		
		perror("[ERROR] size maatrix");
		exit(11);
	}
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] + rigth_part.matrix[i][j];
  return out;
}

mass_th operator-(const mass_th& rigth_part) const
{
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] - rigth_part.matrix[i][j];
  return out;
}


mass_th operator+(const double& x) const
{
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] + x;
  return out;
}

mass_th operator-(const double& x) const
{
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] - x;
  return out;
}

mass_th operator--(int)
{
  mass_th out(col, row);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[j][i] = matrix[i][j];
  return out;
}
void operator*=(const double &x ) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] *= x;
}

void operator/=(const double &x ) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] /= x;
}
void  operator-=( const mass_th &other) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] -= other.matrix[i][j];
}
void  operator-=( const double&other) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] -= other;
}

void operator+=(const mass_th &other) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] += other.matrix[i][j];
}
void operator+=(const double &other) const
{
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] += other;
}
mass_th operator*(const double& x) const
{
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] * x;
  return out;
}

mass_th operator/(const double& x) const
{
  mass_th out(row, col);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[i][j] = matrix[i][j] / x;
  return out;
}

mass_th operator*(const mass_th& rigth_part) const
{

	if (col != rigth_part.row)
	{
		perror("[ERROR] size array");
		exit(11);
	}
  mass_th out(row, rigth_part.col);
   // out.zero();
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < rigth_part.col; j++)
    {
      out.matrix[i][j] = 0;
      for (int r = 0; r < col; r++)
        out.matrix[i][j] += matrix[i][r] * rigth_part.matrix[r][j];
    }
  return out;
}
  void display_m();
	void random();
  void random(double start, double stop);
  void eye();
  void ones();
  void zero();
  double ret(int i, int j);
  mass_th tr();

};

#endif
