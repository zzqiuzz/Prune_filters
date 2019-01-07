#include<gflags/gflags.h>
#include<glog/logging.h>
#include<string>
#include<vector>
#include<map>
#include<caffe/caffe.hpp>
#include<direct.h>
#include<stdio.h>
#include<fstream>
//#define specified_layer
using namespace caffe;
using caffe::string;
/*
* @brief : save parameters for all layers or specified layers 
* @author: zz
* save all layer parameters: comment string target_layer_name; 
* save specified layer parameters : uncomment string target_layer_name;
*
* Notice! Every time you save parameters,the model you take must be the finetuned one!
*/
int main(int argc, char** argv)
{
	//AlexNet.command: G:\caffe\examples\imagenet\deploy.prototxt G:\caffe\models\bvlc_alexnet\caffe_alexnet_model
	//VGG16.command: G:\caffe\models\VGG16\VGG_ILSVRC_16_layers_deploy_orig.prototxt G:\caffe\models\VGG16\VGG_ILSVRC_16_layers.caffemodel
	//G:\caffe\models\VGG16\VGG_test.prototxt G:\caffe\models\VGG16\EA_pruned_conv3_2_128_ft.caffemodel
	//modify target_layer_name;
	//G:\caffe\models\VGG16\AlexNet-BN.prototxt G:\caffe\models\VGG16\final_53.604_77.01_direct_update.caffemodel
	//allkill:G:\caffe\models\VGG16\allkill\VGG_ILSVRC_16_layers_deploy.prototxt G:\caffe\models\VGG16\allkill\VGG_ILSVRC_16_layers.caffemodel
	//G:\caffe\models\VGG16\Mini_VGG_gap_b.prototxt G:\caffe\models\VGG16\Mini_VGG_gap_solver_b_iter_220000.caffemodel
	FLAGS_alsologtostderr = 1;
	caffe::GlobalInit(&argc, &argv);

	//Initialize the network.
	string model_framework_file = argv[1];
	string weights_file = argv[2]; 
	string target_layer_name = "";
	string directory = "";
	string filePath = "";
	//specified layer parameters
	#ifdef specified_layer 
	target_layer_name = "conv4_1";//
	directory = "G:/Matlab/GA_prunning_filters_sample_1W/data/"+target_layer_name+"_param/";
	filePath = "";
	mkdir(directory.c_str()); 
	#endif // 

	
	
	
	shared_ptr<Net<float> > net_;
	net_.reset(new Net<float>(model_framework_file, TEST));
	NetParameter param;
	ReadNetParamsFromBinaryFileOrDie(weights_file,&param);

	int num_layers = param.layer_size();
	const vector<shared_ptr<Layer<float> > > layers = net_->layers();
	const vector<string> layer_names = net_->layer_names();
	for (int i = 0; i < num_layers; i++)//save parameters without bias
	{
		const LayerParameter& source_layer = param.layer(i);
		const string& layer_name = source_layer.name();
		#ifdef specified_layer
			if (layer_name != target_layer_name)continue;
		#endif
		
		int target_layer_id = 0;
		while (target_layer_id != layer_names.size() &&
			layer_names[target_layer_id] != layer_name) {
			++target_layer_id;
		}
		if (target_layer_id == layer_names.size()) {
			LOG(INFO) << "Ignoring source layer " << layer_name;
			continue;
		}
		LOG(INFO) << "Now we come to " << layer_name << " layer";
		vector<shared_ptr<Blob<float> > >& save_blobs =
			layers[target_layer_id]->blobs();
		//filePath = directory+layer_name + ".bin";
		for (int j = 0; j < save_blobs.size(); ++j)
		{
			if (j == 0)
				filePath = directory + layer_name + "weight.bin";
			else
				filePath = directory + layer_name + "bias.bin";
			const bool kReshape = false;
			save_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
			const float* datapoint = save_blobs[j]->cpu_data();
			LOG(INFO) << save_blobs[j]->shape_string();
			std::cout << "Before saving files, print the first data: " << datapoint[0] << std::endl;
			/*for (int start = 0; start < 11 * 11 * 3; start++)
				std::cout << datapoint[start] << std::endl;*/
			//write data to bin file
			int rtnVal = 0;
			FILE* outFile = fopen(filePath.c_str(), "wb");
			rtnVal = fwrite(datapoint, sizeof(float), save_blobs[j]->count(), outFile);
			CHECK_EQ(rtnVal, save_blobs[j]->count()) << "ERROR!";
			fclose(outFile);

			//read data from bin file
			std::cout << "After saving files,read the first data from saved files:";
			rtnVal = 0;
			float data1 = 0;
			FILE* inFile = fopen(filePath.c_str(), "rb");
			rtnVal = fread(&data1, sizeof(float), 1, inFile);
			std::cout << data1 << " " << std::endl;
			fclose(inFile);

			LOG(INFO) << layer_name<<" parameters saved!";  
			
		}

		if (layer_name == target_layer_name)break;


	}

	system("pause");
	return 1;
}


