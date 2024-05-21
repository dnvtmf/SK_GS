#include "common.hpp"

vector<pybind11_init_fp>& PyTrochExtenstionRegistry::Registry() {
  static vector<pybind11_init_fp>* g_registry_ = new vector<pybind11_init_fp>();
  return *g_registry_;
}
void PyTrochExtenstionRegistry::AddExtension(pybind11_init_fp f) {
  vector<pybind11_init_fp>& g_registry_ = Registry();
  g_registry_.push_back(f);
}
void PyTrochExtenstionRegistry::InitAllExtension(py::module_& m) {
  vector<pybind11_init_fp>& g_registry_ = Registry();
  for (auto& fp : g_registry_) fp(m);
  // printf("regist %d functions.\n", (int) g_registry_.size());
};
PyTorchExtensionRegistrer::PyTorchExtensionRegistrer(pybind11_init_fp f) { PyTrochExtenstionRegistry::AddExtension(f); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { PyTrochExtenstionRegistry::InitAllExtension(m); }
