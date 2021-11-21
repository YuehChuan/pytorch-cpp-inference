#include "../infer.h"
#include "crow_all.h"
#include "base64.h"
#include "json11.hpp"
#include "assert.h"
#include <algorithm>
int PORT = 8180;

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





    class Point {
    public:
        int x;
        int y;
        Point (int x, int y) : x(x), y(y) {}
        json11::Json to_json() const { return json11::Json::array { x, y }; }
    };

    class Pattern {

    public:
        int ch;
        //std::vector<std::string> base64Pattern;
        std::string base64Pattern[12];
        cv::Mat imageList[12];
        cv::Mat scrollImage;
        Pattern () {}

    };

    CROW_ROUTE(app, "/jsonPost").methods("POST"_method, "GET"_method)
            ([&image_height, &image_width,
                     &mean, &std, &labels, &model, &usegpu](const crow::request& req){
                std::ostringstream os;


                try {

                    // https://crowcpp.org/reference/classcrow_1_1json_1_1rvalue.html#a2b938dacf1809bb38add4ac8bbeb46ed
                    //https://github.com/dropbox/json11/issues/84
                    //https://github.com/dropbox/json11/issues/106
                    //https://stackoverflow.com/questions/12702561/iterate-through-a-c-vector-using-a-for-loop
                    //https://stackoverflow.com/questions/14373934/iterator-loop-vs-index-loop
                    std::string input_error;
                    auto args= json11::Json::parse(req.body, input_error);
                    if(args == nullptr)
                        crow::response(400);

                    std::vector<std::string>base64RawList;
                    std::vector<Pattern> PRPDArrays;

/*
                    std::vector<json11::Json> v;
                    v=args["JSONAiInputs"].array_items();

                    for (std::size_t i = 0; i != v.size(); ++i) {
                        int ch = v[i]["ch"].int_value();
                        std::cout << ch << std::endl;
                        Pattern P;
                        P.ch=ch;
                        auto k=v[i]["PRPDArrays"].array_items();
                        P.base64Pattern.push_back(k.dump());
                    }
*/

                    std::vector<json11::Json> v;
                    v=args["JSONAiInputs"].array_items();
                    /*get channel*/
                    for (std::size_t i = 0; i != v.size(); ++i) {
                        int ch = v[i]["ch"].int_value();
                        std::cout << ch << std::endl;
                        Pattern P;
                        P.ch=ch;
                        PRPDArrays.push_back(P);
                    }

                    /*get all patterns*/
                    for(auto& j: args["JSONAiInputs"].array_items()) {
                        for(auto& k: j["PRPDArrays"].array_items()){
                            //auto val = k.string_value(); // It can be int_value(), number_value() too.
                            base64RawList.push_back(k.string_value());
                        }
                    }


                    //for(int i = 0; i < base64RawList.size(); i++)
                    //{
                    //    std::cout << base64RawList[i] << std::endl;
                    //}
                    int n=base64RawList.size();
                    std::cout<< n ;
                    assert( (n%12)==0);

                    int k=0; int j=0;
                    for(int i = n - 1; i >= 0; i--) //notice vector elements order related to send Time
                    {
                        if(i==n-1)//i==23
                        {
                            PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            j++;
                            continue;
                        }
                        
                        if( ( (i+1)%12 )!=0)
                        {
                            PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            j++;
                        }
                        else
                        {
                            k++;
                            PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            j++;
                        }

                    }
                    for(int i = 0; i < PRPDArrays.size(); i++)
                    {
                        for (std::size_t j = 0; j != 12; j++) {
                            std::string ss=PRPDArrays[i].base64Pattern[j];
                            std::cout<<(ss)<< std::endl;;
                        }
                    }
                    std::vector<Point> points = { { 1, 2 }, { 10, 20 }, { 100, 200 } };
                    std::string points_json = json11::Json(points).dump();
                    return crow::response(points_json);


                } catch (std::exception& e){

                    std::vector<Point> points = { { 1, 2 }, { 10, 20 }, { 100, 200 } };
                    std::string points_json = json11::Json(points).dump();
                    return crow::response(points_json);
                }

            });

//https://blog.csdn.net/qq_30214939/article/details/70240302
    CROW_ROUTE(app, "/Analysis/By/P").methods("POST"_method, "GET"_method)
            ([&image_height, &image_width,
                     &mean, &std, &labels, &model, &usegpu](const crow::request& req){
                std::ostringstream os;

                try {

                    std::string input_error;
                    auto args= json11::Json::parse(req.body, input_error);
                    if(args == nullptr)
                        crow::response(400);

                    std::vector<std::string>base64RawList;
                    std::vector<cv::Mat> imageRawList;
                    std::vector<Pattern> PRPDArrays;

                    std::vector<json11::Json> v;
                    v=args["JSONAiInputs"].array_items();
                    /*get channel*/
                    for (std::size_t i = 0; i != v.size(); ++i) {
                        int ch = v[i]["ch"].int_value();
                        std::cout << ch << std::endl;
                        Pattern P;
                        P.ch=ch;
                        PRPDArrays.push_back(P);
                    }

                    /*get all patterns*/
                    for(auto& j: args["JSONAiInputs"].array_items()) {
                        for(auto& k: j["PRPDArrays"].array_items()){
                            //base64RawList.push_back(k.dump());
                            base64RawList.push_back(k.string_value());
                            std::string decoded_image = base64_decode(k.string_value());
                            std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
                            cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
                            imageRawList.push_back(image.clone());
                        }
                    }

                    std::reverse(imageRawList.begin(), imageRawList.end());//the image order and why


                    int n=base64RawList.size();
                    std::cout<< n ;
                    assert( (n%12)==0);

                    int k=0; int j=0;
                    for(int i = n - 1; i >= 0; i--) //notice vector elements order related to send Time
                    {
                        if(i==n-1)//i==23
                        {
                            //PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            PRPDArrays[k].imageList[j%12]=imageRawList[i];
                            j++;
                            continue;
                        }

                        if( ( (i+1)%12 )!=0)
                        {
                            //PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            PRPDArrays[k].imageList[j%12]=imageRawList[i];
                            j++;
                        }
                        else
                        {
                            k++;
                            //PRPDArrays[k].base64Pattern[j%12]=base64RawList[i];
                            PRPDArrays[k].imageList[j%12]=imageRawList[i];
                            j++;
                        }

                    }
//https://cloud.tencent.com/developer/article/1697509
                    for(int i = 0; i < PRPDArrays.size(); i++)
                    {
                        std::vector<cv::Mat> matArray;
                        for (std::size_t j = 0; j != 12; j++) {
                           // std::string ss=PRPDArrays[i].base64Pattern[j];
                            matArray.push_back( PRPDArrays[i].imageList[j] );
                           // std::cout<<(ss)<< std::endl;;
                        }
                        cv::hconcat( matArray, PRPDArrays[i].scrollImage );
                        // Predict
                        std::string pred, prob;
                        tie(pred, prob) = infer(PRPDArrays[i].scrollImage, image_height, image_width, mean, std, labels, model, usegpu);
                        std::cout << prob << std::endl;

                        matArray.clear();
                        /*
                        std::string path= "/home/schwarm/pytorch-cpp-inference/inference-cpp/cnn-classification/server/test/data/result/.png";
                        int l=path.length();
                        path.insert(l-4,std::to_string(i));
                        std::cout<< path<<std::endl;
                        cv::imwrite(path, PRPDArrays[i].scrollImage);
                        */

                    }

                    std::vector<Point> points = { { 1, 2 }, { 10, 20 }, { 100, 200 } };
                    std::string points_json = json11::Json(points).dump();
                    return crow::response(points_json);


                } catch (std::exception& e){

                    std::vector<Point> points = { { 1, 2 }, { 10, 20 }, { 100, 200 } };
                    std::string points_json = json11::Json(points).dump();
                    return crow::response(points_json);
                }

            });



  app.port(PORT).run();
  return 0;
}



