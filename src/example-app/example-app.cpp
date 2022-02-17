#include <torch/script.h> // One-stop header.
// #include <torch/all.h>
// #include <soundio/soundio.h>

#include <iostream>
#include <memory>

//https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#executing-programs

// sugar for the tuple access syntax
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
  torch::jit::IValue h, x;
  h = torch::zeros({1, 128}); // TODO: get h0 out of model
  x = torch::zeros({1, 1});

  auto inputs = std::vector<torch::jit::IValue>(3);

  // warm-up for timing
  inputs[0] = h;
  inputs[1] = x;
  inputs[2] = torch::rand({1, 1});
  auto outputs = module.forward(inputs);
  // auto staticmodule = StaticModule(module);
  // auto outputs = staticmodule(inputs);

  float wave[8000];
  const int block_size = 64;

  for (int block=0; block<4096; block++){
    auto start = std::chrono::steady_clock::now();
    for (int i=0; i<block_size; i++){
      // std::vector<torch::jit::IValue> inputs = {h, x, u};
      
      inputs[0] = h;
      inputs[1] = x;
      inputs[2] = torch::rand({1, 1});

      outputs = module.forward(inputs);
      // auto outputs = staticmodule(inputs);

      h = getitem(outputs, 0);//.toTensor();
      x = getitem(outputs, 1);//.toTensor();

      wave[i] = x.toTensor().item<float>();

      // std::cout << x.toTensor().item<float>() << '\n';
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << block_size/elapsed_seconds.count() << '\n';
  }

}