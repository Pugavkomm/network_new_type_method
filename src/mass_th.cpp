#include "mass_th.h"
mass_th::mass_th(int r, int c)
{
  row = r;
  col = c;
  if (c == 1)

    name = "column";
  else if (r == 1)
    name = "row";
  else
    name = "matrix";
  matrix = new double *[row];
  for (int i = 0; i < row; i++)
  {
    matrix[i] = new double[col];
  }
  //cout << this << "- Constr\n";
}

mass_th::mass_th(int r, int c, string names)
{
  name = names;
  row = r;
  col = c;
  name = names;
  matrix = new double *[row];
  for (int i = 0; i < row; i++)
  {
    matrix[i] = new double[col];
  }
  //cout << this << "- Constr\n";
}
mass_th::mass_th(int r)
{
  //cout << this << "- Construct\n";
  row = r;
  col = 1;
  name = "collumn";
  matrix = new double *[row];
  for (int i = 0; i < row; i++)
  {
    matrix[i] = new double[col];
  }
  //cout << this << "- Constr\n";
}
mass_th::mass_th(int r, string names)
{
  row = r;
  col = 1;
  name = names;
  name = "collumn";
  matrix = new double *[row];
  for (int i = 0; i < row; i++)
  {
    matrix[i] = new double[col];
  }
  //cout << this << "- Constr\n";
}
void mass_th::display_m()
{
  cout << name << "\n";
  if (row < 20 && col < 20)
  {
    for (int i = 0; i < col + 1; i++)
      cout << "- ";
    cout << '\n';
    for (int i = 0; i < row; i++)
    {
      for (int j = 0; j < col; j++)
      {
        if (j == 0)
          cout << "|";

        //cout << std::scientific<<
        cout << matrix[i][j];
        if (j < col - 1)
          cout << ' ';
        if (j == col - 1)
        {
          cout << "|";
          if (i == row / 2)
            cout << ", (row = " << row << ", col = " << col << ')';
        }
      }
      cout << '\n';
    }
    for (int i = 0; i < col + 1; i++)
      cout << "- ";
    cout << '\n';
  }
}

void mass_th::random(double start, double stop)
{
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] = start + (stop - start) * (double)rand() / RAND_MAX;
}

void mass_th::random()
{
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] = (double)rand() / RAND_MAX;
}
void mass_th::ones()
{
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] = 1.0;
}

void mass_th::zero()
{
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      matrix[i][j] = 0.0;
}
void mass_th::eye()
{
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      if (i == j)
        matrix[i][j] = 1.0;
      else
        matrix[i][j] = 0.0;
}
double mass_th::ret(int i, int j)
{
  return matrix[i][j];
}

mass_th mass_th::tr()
{
  mass_th out(col, row);
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      out.matrix[j][i] = matrix[i][j];
  return out;
}