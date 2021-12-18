#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

// sugar for the appalling tuple access syntax...
torch::jit::IValue getitem(torch::jit::IValue v, int i) {
  return v.toTuple()->elements()[i];
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  c10::InferenceMode guard;

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  // Create a vector of inputs.
  torch::jit::IValue h, x, u;
  h = torch::zeros({1, 128}); // TODO: get h0 out of model
  x = torch::zeros({1, 1});

  for (int i=0; i<64; i++){
    u = torch::rand({1, 1});
    std::vector<torch::jit::IValue> inputs = {h, x, u};

    // Execute the model and turn its output into a tensor.
    auto outputs = module.forward(inputs);
    h = getitem(outputs, 0);//.toTensor();
    x = getitem(outputs, 1);//.toTensor();

    std::cout << x.toTensor().item<float>() << '\n';
  }

}