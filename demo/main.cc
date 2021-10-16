#include <iostream>

#include "src/include/public.hpp"

Eigen::MatrixXd a, b, c;

void Init(int n, int d, int m) {
  a.resize(n, d);
  b.resize(d, m);
  a.setRandom();
  b.setRandom();
  c = a * b;
}

void KMeans(const Eigen::MatrixXd& a, int num_centers, int num_iters,
            std::vector<VectorXd>* centers, std::vector<int>* belongs) {
  int n = a.rows(), d = a.cols();
  belongs->resize(n);
  centers->resize(num_centers);
  std::vector<int> tot(num_centers);
  for (int i = 0; i < num_centers; i++) {
    belongs->at(i) = i;
    centers->at(i).resize(d);
  }
  for (int i = num_centers; i < n; i++) {
    belongs->at(i) = rand() % num_centers;
  }
  while (num_iters--) {
    for (int i = 0; i < num_centers; i++) {
      centers->at(i).setZero();
      tot[i] = 0;
    }

    for (int i = 0; i < n; i++) {
      int idx = belongs->at(i);
      centers->at(idx) += a.row(i).transpose();
      tot[idx] += 1;
    }
    for (int i = 0; i < num_centers; i++) {
      centers->at(i) /= tot[i];
    }
    for (int i = 0; i < n; i++) {
      int best = -1;
      double best_dis;
      for (int j = 0; j < num_centers; j++) {
        double dis = (centers->at(j) - a.row(i).transpose()).norm();
        if (best < 0 || dis < best_dis) {
          best = j;
          best_dis = dis;
        }
      }
      belongs->at(i) = best;
    }
  }
}

void BoltMM() { BoltEncoder enc(32); }

int main() {
  Init(16, 2, 1);

  std::vector<VectorXd> centers;
  std::vector<int> belongs;
  KMeans(a, 4, 32, &centers, &belongs);

  for (int i = 0; i < a.rows(); i++) {
    std::cout << belongs[i] << ": " << a.row(i) << "\n";
  }
  std::cout << "\n";

  for (int i = 0; i < centers.size(); i++) {
    std::cout << centers[i].transpose() << "\n";
  }
  return 0;
}