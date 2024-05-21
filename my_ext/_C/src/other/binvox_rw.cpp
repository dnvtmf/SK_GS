/*
reference doc: https://www.patrickmin.com/binvox/binvox.html
Reference code: https://www.patrickmin.com/binvox/read_binvox.html
*/

#include <fstream>
#include <iostream>
#include <string>

#include "common.hpp"

using namespace std;

vector<Tensor> binvox_read(string filespec, bool fix_coords = true) {
  ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);

  //
  // read header
  //
  Tensor voxels, translate, scale;
  string line;
  *input >> line;  // #binvox
  if (line.compare("#binvox") != 0) {
    BCNN_ASSERT(false, "Error: first line reads [", line, "] instead of [#binvox]");
    delete input;
    return {voxels, translate, scale};
  }
  int version;
  *input >> version;
  // cout << "reading binvox version " << version << endl;

  int depth, height, width;
  depth    = -1;
  int done = 0;
  while (input->good() && !done) {
    *input >> line;
    if (line.compare("data") == 0)
      done = 1;
    else if (line.compare("dim") == 0) {
      *input >> depth >> height >> width;
    } else if (line.compare("translate") == 0) {
      translate = at::zeros(3, torch::kFloat64);
      double *t = translate.data<double>();
      *input >> t[0] >> t[1] >> t[2];
    } else if (line.compare("scale") == 0) {
      scale     = at::zeros(1, torch::kFloat64);
      double *s = scale.data<double>();
      *input >> s[0];
    } else {
      cerr << "Reading binvox: unrecognized keyword [" << line << "], skipping" << endl;
      char c;
      do {  // skip until end of line
        c = input->get();
      } while (input->good() && (c != '\n'));
    }
  }
  if (!done) {
    BCNN_ASSERT(false, "error reading header");
    return {voxels, translate, scale};
  }
  if (depth == -1) {
    BCNN_ASSERT(false, "missing dimensions in header");
    return {voxels, translate, scale};
  }

  voxels         = torch::empty({depth, height, width}, torch::kUInt8);
  uint8_t *v_ptr = voxels.data<uint8_t>();
  int size       = width * height * depth;

  //
  // read voxel data
  //
  uint8_t value, count;
  int index     = 0;
  int end_index = 0;
  int nr_voxels = 0;

  input->unsetf(ios::skipws);  // need to read every byte now (!)
  *input >> value;             // read the linefeed char

  while ((end_index < size) && input->good()) {
    *input >> value >> count;

    if (input->good()) {
      end_index = index + count;
      if (end_index > size) {
        BCNN_ASSERT(false, "Error binvox file");
      }
      for (int i = index; i < end_index; i++) v_ptr[i] = value;

      if (value) nr_voxels += count;
      index = end_index;
    }  // if file still ok

  }  // while

  input->close();
  if (fix_coords) voxels = voxels.permute({0, 2, 1});

  return {voxels, translate, scale};
}

vector<Tensor> binvox_read_2(string filename, bool fix_coords = true) {
  const int BUFFER_SIZE = 1024 * 1024;
  uint8_t buffer[BUFFER_SIZE];
  FILE *f = fopen(filename.data(), "rb");
  int len, i, j = 0;
  int state = 0, key_type = 0;
  string key = "", value = "";
  uint8_t v = 0;
  uint8_t *data_ptr;
  vector<int> dims;
  vector<double> translate, scale;
  Tensor voxels;

  while ((len = fread(buffer, 1, BUFFER_SIZE, f)) > 0) {
    for (i = 0; i < len; ++i) {
      switch (state) {
        case 0:  // read #binvox
          if (buffer[i] == (uint8_t) ' ') {
            BCNN_ASSERT(key.compare(string("#binvox")) == 0, "not a valid binvox file");
            key   = "";
            state = 1;
          } else
            key += char(buffer[i]);
          break;
        case 1:  // read version, ignored
          if (buffer[i] == '\n') state = 2;
          break;
        case 2:  // read key
          if (buffer[i] == uint8_t(' ') || buffer[i] == uint8_t('\n')) {
            if (key.compare(string("dim")) == 0)
              state = 3, key_type = 1;
            else if (key.compare(string("translate")) == 0)
              state = 3, key_type = 2;
            else if (key.compare(string("scale")) == 0)
              state = 3, key_type = 3;
            else if (key.compare(string("data")) == 0) {
              state = 4;
              j     = 0;
              BCNN_ASSERT(dims.size() == 3, "not a valid binvox file, data");
              voxels   = torch::empty({dims[0], dims[1], dims[2]}, torch::kUInt8);
              data_ptr = voxels.data<uint8_t>();
            } else
              state = 3, key_type = 0;
            key = "";
          } else
            key += char(buffer[i]);
          break;
        case 3:  // read value
          if (buffer[i] == uint8_t(' ') || buffer[i] == uint8_t('\n')) {
            if (key_type == 1)
              dims.push_back(std::stoi(value));
            else if (key_type == 2)
              translate.push_back(std::stod(value));
            else if (key_type == 3)
              scale.push_back(std::stod(value));
            value.clear();
            if (buffer[i] == uint8_t('\n')) state = 2;
          } else {
            value += char(buffer[i]);
          }
          break;
        case 4:
          if (j == 0) {
            v = buffer[i];
          } else {
            int cnt = buffer[i];
            for (; cnt > 0; cnt--) {
              *(data_ptr++) = v;
            }
          }
          j ^= 1;
          break;
        default: break;
      }
    }
  }
  BCNN_ASSERT(state == 4 && j == 0, "not a valid binvox file, end");
  fclose(f);
  if (fix_coords) voxels = voxels.permute({0, 2, 1});
  Tensor t = torch::from_blob(translate.data(), {(int32_t) translate.size()}, torch::kFloat64).clone();
  Tensor s = torch::from_blob(scale.data(), {(int32_t) scale.size()}, torch::kFloat64).clone();
  return {voxels, t, s};
}

vector<Tensor> binvox_read_sparse(string filename, bool fix_coords = true) {
  const int BUFFER_SIZE = 1024 * 1024;
  uint8_t buffer[BUFFER_SIZE];
  FILE *f = fopen(filename.data(), "rb");
  int len, i, j = 0;
  size_t k  = 0;
  int state = 0, key_type = 0;
  string key = "", value = "";
  uint8_t v = 0;
  vector<int> dims;
  vector<float> translate, scale;
  int x, y, z, H = 1, W = 1;

  vector<int> coords;

  while ((len = fread(buffer, 1, BUFFER_SIZE, f)) > 0) {
    for (i = 0; i < len; ++i) {
      switch (state) {
        case 0:  // read #binvox
          if (buffer[i] == (uint8_t) ' ') {
            BCNN_ASSERT(key.compare(string("#binvox")) == 0, "not a valid binvox file");
            key   = "";
            state = 1;
          } else
            key += char(buffer[i]);
          break;
        case 1:  // read version, ignored
          if (buffer[i] == '\n') state = 2;
          break;
        case 2:  // read key
          if (buffer[i] == uint8_t(' ') || buffer[i] == uint8_t('\n')) {
            if (key.compare(string("dim")) == 0)
              state = 3, key_type = 1;
            else if (key.compare(string("translate")) == 0)
              state = 3, key_type = 2;
            else if (key.compare(string("scale")) == 0)
              state = 3, key_type = 3;
            else if (key.compare(string("data")) == 0) {
              state = 4;
              j     = 0;
              k     = 0;
              BCNN_ASSERT(dims.size() == 3, "not a valid binvox file, data");
              // D = dims[0];
              H = dims[1];
              W = dims[2];
            } else
              state = 3, key_type = 0;
            key = "";
          } else
            key += char(buffer[i]);
          break;
        case 3:  // read value
          if (buffer[i] == uint8_t(' ') || buffer[i] == uint8_t('\n')) {
            if (key_type == 1)
              dims.push_back(std::stoi(value));
            else if (key_type == 2)
              translate.push_back(std::stof(value));
            else if (key_type == 3)
              scale.push_back(std::stof(value));
            value.clear();
            if (buffer[i] == uint8_t('\n')) state = 2;
          } else {
            value += char(buffer[i]);
          }
          break;
        case 4:
          if (j == 0) {
            v = buffer[i];
          } else {
            int cnt = buffer[i];
            if (v == 1) {
              for (; cnt > 0; cnt--, k++) {
                size_t t = k / W;
                y        = k % W;
                z        = t % H;
                x        = t / H;
                if (fix_coords) {
                  coords.push_back(x);
                  coords.push_back(y);
                  coords.push_back(z);
                } else {
                  coords.push_back(x);
                  coords.push_back(z);
                  coords.push_back(y);
                }
              }
            } else {  // v == 0
              k += cnt;
            }
          }
          j ^= 1;
          break;
        default: break;
      }
    }
  }
  BCNN_ASSERT(state == 4 && j == 0, "not a valid binvox file, end");
  fclose(f);

  return {torch::from_blob(coords.data(), {(int32_t) coords.size() / 3, 3}, torch::kInt32).to(torch::kInt64),
      torch::from_blob(dims.data(), {(int32_t) dims.size()}, torch::kInt32).to(torch::kInt64),
      torch::from_blob(translate.data(), {(int32_t) translate.size()}, torch::kFloat32).clone(),
      torch::from_blob(scale.data(), {(int32_t) scale.size()}, torch::kFloat32).clone()};
}

void write_binvox(string filename, Tensor &voxels, Tensor &translate, Tensor &scale);

REGIST_PYTORCH_EXTENSION(binvox_rw, {
  m.def("binvox_read", &binvox_read_2, "Read BinVox", py::arg("filename"), py::arg("fix_coords") = true);
  m.def("binvox_read_sparse", &binvox_read_sparse, "Read BinVox to coordations", py::arg("filename"),
      py::arg("fix_coords") = true);
})
