#include <glog/logging.h>

#include <iostream>
#include <memory>

#include "src/include/public.hpp"
#include "unsupported/Eigen/MatrixFunctions"

#define MAP_MAT(mat)                                                  \
  std::vector<float> mat##_data(mat.rows() * mat.cols());             \
  do {                                                                \
    Matrix<float, Dynamic, Dynamic, RowMajor> tmp_mat(mat);           \
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(                   \
        mat##_data.data(), tmp_mat.rows(), tmp_mat.cols()) = tmp_mat; \
  } while (0)

void Init(int n, int d, int m, MatrixXf* a, MatrixXf* b, MatrixXf* c) {
  a->resize(n, d);
  b->resize(d, m);
  a->setRandom();
  b->setRandom();
  *c = *a * *b;
}

void KMeans(const MatrixXf& a, int num_centroids, int num_iters,
            MatrixXf* centroids, std::vector<int>* belongs) {
  int n = a.rows(), d = a.cols();
  belongs->resize(n);
  centroids->resize(num_centroids, d);
  std::vector<int> tot(num_centroids);
  for (int i = 0; i < n; i++) {
    belongs->at(i) = rand() % num_centroids;
  }
  while (num_iters--) {
    centroids->setZero();
    for (int i = 0; i < num_centroids; i++) {
      tot[i] = 0;
    }

    for (int i = 0; i < n; i++) {
      int idx = belongs->at(i);
      centroids->row(idx) += a.row(i);
      tot[idx] += 1;
    }
    for (int i = 0; i < num_centroids; i++) {
      if (tot[i] > 0) {
        centroids->row(i) /= tot[i];
      } else {
        centroids->row(i).setRandom();
      }
    }
    for (int i = 0; i < n; i++) {
      int best = -1;
      double best_dis;
      for (int j = 0; j < num_centroids; j++) {
        double dis = (centroids->row(j) - a.row(i)).norm();
        if (best < 0 || dis < best_dis) {
          best = j;
          best_dis = dis;
        }
      }
      belongs->at(i) = best;
    }
  }
}

VectorXf Percentile(const MatrixXf& mat, float percentage) {
  CHECK(0 <= percentage && percentage <= 1);
  int idx = std::min(int(percentage * mat.rows()), (int)(mat.rows() - 1));
  float ratio = std::min(percentage * mat.rows() - idx, 1.0f);
  VectorXf ret(mat.cols());
  for (int i = 0; i < mat.cols(); i++) {
    VectorXf vec = mat.col(i);
    std::sort(vec.data(), vec.data() + vec.size());
    if (idx < mat.rows() - 1) {
      ret[i] = (1 - ratio) * vec[idx] + ratio * vec[idx + 1];
    } else {
      ret[i] = vec[idx];
    }
  }
  return ret;
}

float PercentileFloat(const MatrixXf& mat, float percentage) {
  CHECK(0 <= percentage && percentage <= 1);
  MAP_MAT(mat);
  std::sort(mat_data.begin(), mat_data.end());
  int idx =
      std::min(int(percentage * mat_data.size()), (int)(mat_data.size() - 1));
  float ratio = std::min(percentage * mat_data.size() - idx, 1.0f);
  float ret;
  if (idx < mat_data.size() - 1) {
    ret = (1 - ratio) * mat_data[idx] + ratio * mat_data[idx + 1];
  } else {
    ret = mat_data[idx];
  }
  return ret;
}

void CalcScaleAndOffsets(const MatrixXf& fake_b_t,
                         const std::vector<MatrixXf>& codebooks,
                         VectorXf* out_floor, float* out_scale_by,
                         float* out_alpha) {
  int subvect_len = fake_b_t.cols() / codebooks.size();
  MatrixXf luts(fake_b_t.rows() * 16, codebooks.size());
  for (int i = 0; i < fake_b_t.rows(); i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < codebooks.size(); k++) {
        RowVector<float> tmp =
            fake_b_t.block(i, k * subvect_len, 1, subvect_len);
        luts(i * 16 + j, k) = tmp.dot(codebooks[k].row(j));
      }
    }
  }

  float best_loss = INFINITY;

  for (float alpha :
       std::vector<float>{0, .001, .002, .005, .01, .02, .05, .1}) {
    VectorXf floor = Percentile(luts, alpha);
    float ceil = PercentileFloat(luts, 1 - alpha);
    MatrixXf offset =
        (luts.array().rowwise() - floor.transpose().array()).cwiseMax(0);
    float scale_by = 255 / ceil;
    MatrixXf luts_ideal = (offset * scale_by).cwiseMin(255);
    MatrixXi luts_quantized = luts_ideal.template cast<int>();
    MatrixXf diff = (luts_ideal - luts_quantized.template cast<float>());
    float loss = diff.squaredNorm();

    if (loss < best_loss) {
      best_loss = loss;
      *out_floor = floor;
      *out_scale_by = scale_by;
      *out_alpha = alpha;
    }
  }
}

std::unique_ptr<BoltEncoder> TrainBolt(const MatrixXf& mat, int nbytes,
                                       float* out_offset, float* out_scale_by) {
  auto enc = std::make_unique<BoltEncoder>(nbytes);
  int ncodebooks = nbytes * 2;
  CHECK(mat.cols() % ncodebooks == 0);
  int subvect_len = mat.cols() / ncodebooks;

  std::vector<MatrixXf> codebooks(ncodebooks);
  MatrixXf centroids(16 * ncodebooks, subvect_len);
  for (int i = 0; i < ncodebooks; i++) {
    std::vector<int> belongs;
    KMeans(mat.block(0, subvect_len * i, mat.rows(), subvect_len), 16, 16,
           &codebooks[i], &belongs);
    centroids.block(16 * i, 0, 16, subvect_len) = codebooks[i];
  }

  VectorXf floor;
  float alpha, scale_by;
  {
    int num_rows = mat.rows() * 0.25;
    MatrixXf fake_b_t(num_rows, mat.cols());
    for (int i = 0; i < num_rows; i++) {
      int j = rand() % mat.rows();
      fake_b_t.row(i) = mat.row(j);
    }

    CalcScaleAndOffsets(fake_b_t, codebooks, &floor, &scale_by, &alpha);
  }

  enc->set_scale(scale_by);
  VectorXf offsets_to_set = -floor * scale_by;
  enc->set_offsets(offsets_to_set.data(), offsets_to_set.size());

  MAP_MAT(centroids);
  enc->set_centroids(centroids_data.data(), centroids.rows(), centroids.cols());

  MAP_MAT(mat);
  enc->set_data(mat_data.data(), mat.rows(), mat.cols());

  *out_scale_by = scale_by;
  *out_offset = 0;
  for (int i = 0; i < offsets_to_set.size(); i++)
    *out_offset += offsets_to_set[i];

  return enc;
}

MatrixXf MMBolt(BoltEncoder* enc, const MatrixXf& mat, float offset,
                float scale_by) {
  MatrixXf ret;
  for (int i = 0; i < mat.cols(); i++) {
    VectorXf col = mat.col(i);
    RowVector<uint16_t> ans = enc->dot_prods(col.data(), col.size());
    RowVector<float> ans_float = ans.template cast<float>();
    ans_float.array() -= offset;
    ans_float.array() /= scale_by;

    if (i == 0) {
      ret = ans_float.transpose();
    } else {
      ret.conservativeResize(ret.rows(), ret.cols() + 1);
      ret.col(ret.cols() - 1) = ans_float.transpose();
    }
  }

  return ret;
}

int main() {
  MatrixXf a, b, c;

  Init(64, 32, 16, &a, &b, &c);
  float offset, scale_by;

  auto enc = TrainBolt(a, 8, &offset, &scale_by);
  MatrixXf test_c = MMBolt(enc.get(), b, offset, scale_by);

  LOG(INFO) << "c:\n" << c;
  LOG(INFO) << "test_c:\n" << test_c;

  return 0;
}