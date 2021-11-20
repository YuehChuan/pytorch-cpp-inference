#include "../infer.h"
#include "crow_all.h"
#include "base64.h"
#include "json11.hpp"
int PORT = 8181;

//./predict geeks.png ../../models/resnet/pds_cpu.pth ../../models/resnet/labels.txt false
//./predict ../../../models/resnet/pds_cpu.pth ../../../models/resnet/labels.txt false
//https://stackoverflow.com/questions/67119896/refer-to-json-individual-objects-in-cpp-crow
int main(int argc, char **argv) {

  if (argc != 4) {
    std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> \n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string labels_path = argv[2];
  std::string usegpu_str = argv[3];
  bool usegpu;

  if (usegpu_str == "true") {
      usegpu = true;
  } else {
      usegpu = false;
  }

  // Set image height and width
  int image_height = 224;
  int image_width = 2688;

  // Read labels
  std::vector<std::string> labels;
  std::string label;
  std::ifstream labelsfile (labels_path);
  if (labelsfile.is_open())
  {
    while (getline(labelsfile, label))
    {
      labels.push_back(label);
    }
    labelsfile.close();
  }

  // Define mean and std
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};

  // Load Model
  torch::jit::script::Module model = read_model(model_path, usegpu);

  // App
  crow::SimpleApp app;
  CROW_ROUTE(app, "/predict").methods("POST"_method, "GET"_method)
  ([&image_height, &image_width,
    &mean, &std, &labels, &model, &usegpu](const crow::request& req){
    crow::json::wvalue result;
    result["Prediction"] = "";
    result["Confidence"] = "";
    result["Status"] = "Failed";
    std::ostringstream os;

    try {
      auto args = crow::json::load(req.body);

      // Get Image
      std::string base64_image = args["image"].s();
      std::string decoded_image = base64_decode(base64_image);
      std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
      cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

      // Predict
      std::string pred, prob;
      tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);
      std::cout << prob << std::endl;
      result["Prediction"] = pred;
      result["Confidence"] = prob;
      result["Status"] = "Success";

      os << crow::json::dump(result);
      return crow::response{os.str()};

    } catch (std::exception& e){
      os << crow::json::dump(result);
      return crow::response(os.str());
    }

  });


//https://github.com/ipkn/crow/blob/master/examples/example.cpp#L144
    CROW_ROUTE(app, "/Analysis/By/P").methods("POST"_method, "GET"_method)
            ([&image_height, &image_width,
                     &mean, &std, &labels, &model, &usegpu](const crow::request& req){
                crow::json::wvalue result;
                result["Prediction"] = "";
                result["Confidence"] = "";
                result["Status"] = "Failed";
                std::ostringstream os;

                try {
                    //auto args = crow::json::load(req.body);
//https://crowcpp.org/reference/classcrow_1_1json_1_1rvalue.html#a2b938dacf1809bb38add4ac8bbeb46ed
                    //https://github.com/dropbox/json11/issues/84
                    std::string input_error;
                    auto args= json11::Json::parse(req.body, input_error);
                    if(args == nullptr)
                        crow::response(400);

                    std::string base64_image = args["image"].string_value();

                    std::vector<std::string> inp;
                    for(auto& j: args["PRPDArrays"].array_items()) {
                        auto val = j.string_value(); // It can be int_value(), number_value() too.
                        inp.push_back(j.dump());
                    }

                    for(int i = 0; i < inp.size(); i++)
                    {
                        std::cout << inp[i] << std::endl;
                    }




                    //std::string base64_image = args["image"].s();
                    std::string decoded_image = base64_decode(base64_image);
                    std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
                    cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

                    // Predict
                    std::string pred, prob;
                    tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);
                    std::cout << prob << std::endl;
                    result["Prediction"] = pred;
                    result["Confidence"] = prob;
                    result["Status"] = "Success";

                    os << crow::json::dump(result);
                    return crow::response{os.str()};

                } catch (std::exception& e){
                    os << crow::json::dump(result);
                    return crow::response(os.str());
                }

            });


  app.port(PORT).run();
  return 0;
}
