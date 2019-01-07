#include<gflags/gflags.h>
#include<glog/logging.h>
#include<string>
#include<vector>
#include<map>
#include<caffe/caffe.hpp>

#include<stdio.h>
#include<fstream>
using namespace caffe;
using caffe::string; 
/*
* command: G:\caffe\models\VGG16\VGG_ILSVRC_16_layers_deploy_EA.prototxt G:\caffe\models\VGG16\EA_pruned_conv4_1_256_ft.caffemodel
* @brief: modify string filePath to read index of EA method-pruned filters.
* @time: 2018.2.5
* @author: zz
* modify **********************
*/
int main(int argc, char** argv)
{

	FLAGS_alsologtostderr = 1;
	caffe::GlobalInit(&argc, &argv);

	//Initialize the network.
	string orig_model_framework_file = argv[1];
	string orig_weights_file = argv[2];
	string pruned_one_filter_one_weights_file_savepath = "";
	shared_ptr<Net<float> > orig_net_;
	NetParameter orig_param;
	NetParameter pruned_param;
	orig_net_.reset(new Net<float>(orig_model_framework_file, TEST));
	ReadNetParamsFromBinaryFileOrDie(orig_weights_file, &orig_param);

	//read best channel indexes for prune ******************************************
	string filePath = "EA_pruned_conv4_1filters_by_conv4_2_channel_selection.bin";
	int pruned_filter_num = 256;//filters number to be pruned == numbers of channels to be selected. conv3_1 剪枝129个
	string target_pruned_layer_name = "conv4_1";
	int rtnVal = 0; 
	int* result_data = new int[pruned_filter_num];
	vector<int> pruned_filter_index;
	FILE* inFile = fopen(filePath.c_str(), "rb");
	rtnVal = fread(result_data, sizeof(int), pruned_filter_num, inFile); 
	fclose(inFile);
	for (int i = 0; i < pruned_filter_num; i++)
		pruned_filter_index.push_back(result_data[i] - 1);
	delete[] result_data;

	int num_layers = orig_param.layer_size();
	const vector<string>& layer_names = orig_net_->layer_names();
	const vector<shared_ptr<Layer<float> > > layers_ = orig_net_->layers();
	map<string, vector<int> > name_to_smallest_index;
	map<string, string> former_conv_to_latter_conv;
	string last_conv = "conv1_1";
	string now_conv;
	vector<string> param_layer_index;
	//int pruned_filter_num = 32;//filters number to be pruned
	
	string ext = "_filter_VGG16.caffemodel";
	stringstream s;
	s << pruned_filter_num;
	string pruned_network_name = "EA_source_pruned_one_" + target_pruned_layer_name + "_" + s.str() + ext;
	for (int layer = 0; layer < num_layers; layer++)//read parameters from caffemodel
	{
		const LayerParameter layer_param = orig_param.layer(layer);
		const string layer_name = layer_param.name();
		vector<float>::iterator smallest_value;
		vector<float>::iterator biggest_value;
		int smallest_index = 0;
		int biggest_index = 0;
		LOG(INFO) << "Now we come to " << layer_name << " layer";
		int target_layer_id = 0;
		while (target_layer_id != layer_names.size() && //to achieve consistency among caffemodel and prototxt 
			layer_names[target_layer_id] != layer_name) {
			++target_layer_id;
		}
		if (target_layer_id == layer_names.size()) {
			LOG(INFO) << "Ignoring source layer " << layer_name;
			continue;
		}
		LOG(INFO) << "Copying source layer " << layer_name;
		vector<shared_ptr<Blob<float> > > target_blobs = layers_[target_layer_id]->blobs();//target_layer_id!
		CHECK_EQ(target_blobs.size(), layer_param.blobs_size())//检查caffemodel中参数与定义文件是否一致
			<< "Incompatible number of blobs for layer " << layer_name;
		for (int blob = 0; blob < target_blobs.size(); blob++)
		{
			if (!target_blobs[blob]->ShapeEquals(layer_param.blobs(blob))) {
				Blob<float> source_blob;
				const bool kReshape = true;
				source_blob.FromProto(layer_param.blobs(blob), kReshape);
				LOG(FATAL) << "Cannot copy param " << blob << " weights from layer '"
					<< layer_name << "'; shape mismatch.  Source param shape is "
					<< source_blob.shape_string() << "; target param shape is "
					<< target_blobs[blob]->shape_string() << ". "
					<< "To learn this layer's parameters from scratch rather than "
					<< "copying from a saved net, rename the layer.";
			}
			//const bool kReshape = false; 
			LOG(INFO) << target_blobs[blob]->shape_string();
			const BlobProto& source_proto = layer_param.blobs(blob);
			float* data_vec = target_blobs[blob]->mutable_cpu_data();
			param_layer_index.push_back(layer_name);

			now_conv = layer_name;
			former_conv_to_latter_conv[last_conv] = now_conv;
			last_conv = now_conv;
			if (blob == 0)//all layers weight except for bias.
			{
				LOG(INFO) << "Filter's num is: " << target_blobs[blob]->num() << " channel's number: " << target_blobs[blob]->channels()
					<< " height is: " << target_blobs[blob]->height() << " width is: " << target_blobs[blob]->width();
				int filter_num = target_blobs[blob]->num();
				int filter_channel = target_blobs[blob]->channels();
				int filter_width = target_blobs[blob]->width();
				int filter_height = target_blobs[blob]->height();
				for (int data = 0; data < target_blobs[blob]->count(); data++)
					data_vec[data] = source_proto.data(data);
				data_vec = target_blobs[blob]->mutable_cpu_data();
				vector<float> each_filter_sum(0, filter_num);
				for (int i = 0; i < filter_num; i++)
				{
					float sum_value = 0;
					sum_value = caffe_cpu_asum(filter_channel*filter_height*filter_width, data_vec);
					data_vec += filter_channel*filter_height*filter_width;
					each_filter_sum.push_back(sum_value);
				}
				//print info of each filter in each layer.
				/*vector<float>::iterator iter;
				for (iter = each_filter_sum.begin(); iter != each_filter_sum.end(); iter++)
				std::cout << *iter << " ";*/

				// sort each_filter_sum with index returned
				vector<float> temp_for_sort = each_filter_sum;
				biggest_value = std::max_element(std::begin(each_filter_sum), std::end(each_filter_sum));
				biggest_index = std::distance(std::begin(each_filter_sum), biggest_value);
				for (int i = 0; i < each_filter_sum.size(); i++)
				{
					smallest_value = std::min_element(std::begin(temp_for_sort), std::end(temp_for_sort));
					biggest_value = std::max_element(std::begin(temp_for_sort), std::end(temp_for_sort));
					smallest_index = std::distance(std::begin(temp_for_sort), smallest_value);
					temp_for_sort[smallest_index] = *biggest_value;
					//smallest_filter_eachlayer.push_back(*smallest_value);
					name_to_smallest_index[layer_name];
					name_to_smallest_index[layer_name].push_back(smallest_index);//由大到小  [0]最大 
				}
				name_to_smallest_index[layer_name][each_filter_sum.size() - 1] = biggest_index;
			}
		}
	}
	//save pruned caffemodel
	pruned_param.Clear();
	pruned_param.set_name(pruned_network_name);
	DLOG(INFO) << "Serializing " << layers_.size() << " layers with the smallest filter in the first layer pruned";
	vector<int> prune_next_layer_channel_param;
	for (int pruned_layer = 0; pruned_layer < num_layers; ++pruned_layer)
	{
		LayerParameter* pruned_layer_param = pruned_param.add_layer();

		const LayerParameter& source_layer_param = layers_[pruned_layer]->layer_param();
		const string layer_name = source_layer_param.name();
		pruned_layer_param->Clear();
		pruned_layer_param->CopyFrom(source_layer_param);
		pruned_layer_param->clear_blobs();
		int target_layer_id = 0;
		while (target_layer_id != layer_names.size() &&
			layer_names[target_layer_id] != layer_name) {
			++target_layer_id;
		}
		if (target_layer_id == layer_names.size()) {
			LOG(INFO) << "Ignoring source layer " << layer_name;
			continue;
		}
		vector<shared_ptr<Blob<float> > > source_blobs = layers_[target_layer_id]->blobs();
		vector<int> prune_bias_param;

		for (int blob = 0; blob < source_blobs.size(); blob++)
		{
			BlobProto* target_proto = pruned_layer_param->add_blobs();
			vector<int> blob_shape = source_blobs[blob]->shape();
			target_proto->clear_shape();
			for (int i = 0; i < blob_shape.size(); ++i) {
				target_proto->mutable_shape()->add_dim(blob_shape[i]);
			}
			target_proto->clear_data();
			target_proto->clear_diff();
			const float* data_vec = source_blobs[blob]->cpu_data();
			//prune filter

			if (source_layer_param.name() == target_pruned_layer_name)// prune m filter in conv1_1 layer!
			{

				//prune conv filter
				if (blob == 0)
				{
					//vector<int> temp_pruned_filter;
					target_proto->mutable_shape()->set_dim(0, blob_shape[0] - pruned_filter_num);//update new target blob shape()
					std::cout << target_proto->mutable_shape()->dim(0) << " " << target_proto->mutable_shape()->dim(1) << " " << target_proto->mutable_shape()->dim(2) << " " << target_proto->mutable_shape()->dim(3);


					//for (int i = 0; i < pruned_filter_num; i++)
					//	temp_pruned_filter.push_back(name_to_smallest_index[target_pruned_layer_name][i]);
					//sort(temp_pruned_filter.begin(), temp_pruned_filter.end());//sort pruned filters by ascending order 
					prune_bias_param = pruned_filter_index;
					prune_next_layer_channel_param = pruned_filter_index;
					int filter_element_number_sum = blob_shape[1] * blob_shape[2] * blob_shape[3];
					int data_loc = 0;
					for (int filter_wise = 0; filter_wise < blob_shape[0]; filter_wise++)
					{
						if (pruned_filter_index.size()>0)
						{
							if (filter_wise == pruned_filter_index[0])
								pruned_filter_index.erase(pruned_filter_index.begin());
							else
							{
								for (int each_filter_element = 0; each_filter_element < filter_element_number_sum; each_filter_element++)
								{
									data_loc = filter_wise*filter_element_number_sum + each_filter_element;
									target_proto->add_data(data_vec[data_loc]);

								}
							}
						}
						else
						{
							for (int each_filter_element = 0; each_filter_element < filter_element_number_sum; each_filter_element++)
							{
								data_loc = filter_wise*filter_element_number_sum + each_filter_element;
								target_proto->add_data(data_vec[data_loc]);

							}
						}

					}
				}
				else//prune bias correspondingly
				{

					target_proto->mutable_shape()->set_dim(0, blob_shape[0] - pruned_filter_num);
					std::cout << target_proto->mutable_shape()->dim(0);

					for (int bias_start = 0; bias_start < blob_shape[0]; bias_start++)
					{
						if (prune_bias_param.size() > 0)
						{
							if (bias_start == prune_bias_param[0])
								prune_bias_param.erase(prune_bias_param.begin());
							else
								target_proto->add_data(data_vec[bias_start]);
						}
						else
							target_proto->add_data(data_vec[bias_start]);

					}
				}
			}
			else if (source_layer_param.name() == former_conv_to_latter_conv[target_pruned_layer_name])//if the smallest filter in conv1_1 pruned, then continue prune channel correspondingly in conv1_2
			{
				if (blob == 0)//channel pruning
				{
					vector<int> temp_pruned_channel = prune_next_layer_channel_param;
					int pruned_channel_num = pruned_filter_num;
					target_proto->mutable_shape()->set_dim(1, blob_shape[1] - pruned_channel_num);
					std::cout << target_proto->mutable_shape()->dim(0) << " " << target_proto->mutable_shape()->dim(1) << " " << target_proto->mutable_shape()->dim(2) << " " << target_proto->mutable_shape()->dim(3);

					//int filter_element_number_sum = blob_shape[1] * blob_shape[2] * blob_shape[3];
					int spatial_element_number_sum = blob_shape[2] * blob_shape[3];
					int channel_num = blob_shape[1];
					int data_loc = 0;
					for (int filter_wise = 0; filter_wise < blob_shape[0]; filter_wise++)//for each filter
					{
						for (int channel_wise = 0; channel_wise < blob_shape[1]; channel_wise++)// for each channel
						{
							if (temp_pruned_channel.size()>0)
							{
								if (channel_wise == temp_pruned_channel[0])
									temp_pruned_channel.erase(temp_pruned_channel.begin());
								else
								{
									for (int element_wise = 0; element_wise < spatial_element_number_sum; element_wise++)
									{
										data_loc = (filter_wise*channel_num + channel_wise)*spatial_element_number_sum + element_wise;
										target_proto->add_data(data_vec[data_loc]);
									}
								}
							}
							else
							{
								for (int element_wise = 0; element_wise < spatial_element_number_sum; element_wise++)
								{
									data_loc = (filter_wise*channel_num + channel_wise)*spatial_element_number_sum + element_wise;
									target_proto->add_data(data_vec[data_loc]);
								}
							}
						}
						temp_pruned_channel = prune_next_layer_channel_param;
					}
				}
				else
				{
					for (int j = 0; j < source_blobs[blob]->count(); ++j)
						target_proto->add_data(data_vec[j]);
				}
			}
			else//normal layers
			{
				for (int j = 0; j < source_blobs[blob]->count(); ++j)
					target_proto->add_data(data_vec[j]);
			}
		}
	}
	WriteProtoToBinaryFile(pruned_param, pruned_network_name);
	system("pause");
	return 1;
}




//#include<gflags/gflags.h>
//#include<glog/logging.h>
//#include<string>
//#include<vector>
//#include<map>
//#include<caffe/caffe.hpp>
//
//#include<stdio.h>
//#include<fstream>
//using namespace caffe;
//using caffe::string;
//
//////AlexNet.command: E:\caffe\examples\imagenet\deploy.prototxt E:\caffe\models\bvlc_alexnet\caffe_alexnet_model
//////VGG16.command: E:\caffe\models\VGG16\VGG_ILSVRC_16_layers_deploy_orig.prototxt E:\caffe\models\VGG16\VGG_ILSVRC_16_layers.caffemodel
////E:\caffe\models\VGG16\VGG_ILSVRC_16_layers_deploy.prototxt E:\caffe\models\VGG16\pruned_conv4_2_256_ft.caffemodel
//
//int main(int argc, char** argv)
//{
//
//	FLAGS_alsologtostderr = 1;
//	caffe::GlobalInit(&argc, &argv);
//
//	//Initialize the network.
//	string orig_model_framework_file = argv[1];
//	string orig_weights_file = argv[2];
//	string pruned_one_filter_one_weights_file_savepath = "";
//	shared_ptr<Net<float> > orig_net_;
//	NetParameter orig_param;
//	NetParameter pruned_param;
//	orig_net_.reset(new Net<float>(orig_model_framework_file, TEST));
//	ReadNetParamsFromBinaryFileOrDie(orig_weights_file, &orig_param);
//
//	//read best channel for prune
//	string filePath = "pruned_conv1_1filters_by_conv1_2_channel_selection.bin";
//	int pruned_filter_num = 32;//filters number to be pruned
//	int rtnVal = 0; 
//	int* result_data = new int[pruned_filter_num];
//	vector<int> pruned_filter_index;
//	FILE* inFile = fopen(filePath.c_str(), "rb");
//	rtnVal = fread(result_data, sizeof(int), pruned_filter_num, inFile); 
//	fclose(inFile);
//	for (int i = 0; i < pruned_filter_num; i++)
//		pruned_filter_index.push_back(result_data[i] - 1);
//		//pruned_filter_index.push_back(result_data[i]-1); //difference between Matlab and C++
//	delete[] result_data;
//
//	int num_layers = orig_param.layer_size();
//	const vector<string>& layer_names = orig_net_->layer_names();
//	const vector<shared_ptr<Layer<float> > > layers_ = orig_net_->layers();
//	map<string, string> former_conv_to_latter_conv;
//	string last_conv = "conv1_1";
//	string now_conv;  
//	string target_pruned_layer_name = "conv1_1";
//	string ext = "_filter_VGG16.caffemodel";
//	stringstream s;
//	s << pruned_filter_num;
//	string pruned_network_name = "EA_source_pruned_one_" + target_pruned_layer_name + "_" + s.str() + ext;
//	//get info from orig_caffemodel
//	for (int layer = 0; layer < num_layers; layer++)//read parameters from caffemodel
//	{
//		const LayerParameter layer_param = orig_param.layer(layer);
//		const string layer_name = layer_param.name();
//		vector<float>::iterator smallest_value;
//		vector<float>::iterator biggest_value;
//		int smallest_index = 0;
//		int biggest_index = 0;
//		LOG(INFO) << "Now we come to " << layer_name << " layer";
//		int target_layer_id = 0;
//		while (target_layer_id != layer_names.size() && //to achieve consistency among caffemodel and prototxt 
//			layer_names[target_layer_id] != layer_name) {
//			++target_layer_id;
//		}
//		if (target_layer_id == layer_names.size()) {
//			LOG(INFO) << "Ignoring source layer " << layer_name;
//			continue;
//		}
//		LOG(INFO) << "Copying source layer " << layer_name;
//		vector<shared_ptr<Blob<float> > > target_blobs = layers_[target_layer_id]->blobs();//target_layer_id!
//		CHECK_EQ(target_blobs.size(), layer_param.blobs_size())//检查caffemodel中参数与定义文件是否一致
//			<< "Incompatible number of blobs for layer " << layer_name;
//		for (int blob = 0; blob < target_blobs.size(); blob++)
//		{
//			if (!target_blobs[blob]->ShapeEquals(layer_param.blobs(blob))) {
//				Blob<float> source_blob;
//				const bool kReshape = true;
//				source_blob.FromProto(layer_param.blobs(blob), kReshape);
//				LOG(FATAL) << "Cannot copy param " << blob << " weights from layer '"
//					<< layer_name << "'; shape mismatch.  Source param shape is "
//					<< source_blob.shape_string() << "; target param shape is "
//					<< target_blobs[blob]->shape_string() << ". "
//					<< "To learn this layer's parameters from scratch rather than "
//					<< "copying from a saved net, rename the layer.";
//			}
//			//const bool kReshape = false; 
//			LOG(INFO) << target_blobs[blob]->shape_string();
//			const BlobProto& source_proto = layer_param.blobs(blob);
//			float* data_vec = target_blobs[blob]->mutable_cpu_data();
//			 
//			now_conv = layer_name;
//			former_conv_to_latter_conv[last_conv] = now_conv;
//			last_conv = now_conv;
//		}
//	}
//	//save pruned caffemodel
//	pruned_param.Clear();
//	pruned_param.set_name(pruned_network_name);
//	DLOG(INFO) << "Serializing " << layers_.size() << " layers with the smallest filter in the first layer pruned";
//	vector<int> prune_next_layer_channel_param;
//	for (int pruned_layer = 0; pruned_layer < num_layers; ++pruned_layer)
//	{
//		LayerParameter* pruned_layer_param = pruned_param.add_layer();
//
//		const LayerParameter& source_layer_param = layers_[pruned_layer]->layer_param();
//		const string layer_name = source_layer_param.name();
//		pruned_layer_param->Clear();
//		pruned_layer_param->CopyFrom(source_layer_param);
//		pruned_layer_param->clear_blobs();
//		int target_layer_id = 0;
//		while (target_layer_id != layer_names.size() &&
//			layer_names[target_layer_id] != layer_name) {
//			++target_layer_id;
//		}
//		if (target_layer_id == layer_names.size()) {
//			LOG(INFO) << "Ignoring source layer " << layer_name;
//			continue;
//		}
//		vector<shared_ptr<Blob<float> > > source_blobs = layers_[target_layer_id]->blobs();
//		vector<int> prune_bias_param;
//
//		for (int blob = 0; blob < source_blobs.size(); blob++)
//		{
//			BlobProto* target_proto = pruned_layer_param->add_blobs();
//			vector<int> blob_shape = source_blobs[blob]->shape();
//			target_proto->clear_shape();
//			for (int i = 0; i < blob_shape.size(); ++i) {
//				target_proto->mutable_shape()->add_dim(blob_shape[i]);
//			}
//			target_proto->clear_data();
//			target_proto->clear_diff();
//			const float* data_vec = source_blobs[blob]->cpu_data();
//			//prune filter
//
//			if (source_layer_param.name() == target_pruned_layer_name)// prune m filter in conv1_1 layer!
//			{
//
//				//prune conv filter
//				if (blob == 0)
//				{
//					vector<int> temp_pruned_filter;
//					target_proto->mutable_shape()->set_dim(0, blob_shape[0] - pruned_filter_num);//update new target blob shape()
//					std::cout << target_proto->mutable_shape()->dim(0) << " " << target_proto->mutable_shape()->dim(1) << " " << target_proto->mutable_shape()->dim(2) << " " << target_proto->mutable_shape()->dim(3);
//
//
//					for (int i = 0; i < pruned_filter_num; i++)
//						temp_pruned_filter.push_back(pruned_filter_index[i]);
//					
//					prune_bias_param = temp_pruned_filter;
//					prune_next_layer_channel_param = temp_pruned_filter;
//					int filter_element_number_sum = blob_shape[1] * blob_shape[2] * blob_shape[3];
//					int data_loc = 0;
//					for (int filter_wise = 0; filter_wise < blob_shape[0]; filter_wise++)
//					{
//						if (temp_pruned_filter.size()>0)
//						{
//							if (filter_wise == temp_pruned_filter[0])
//								temp_pruned_filter.erase(temp_pruned_filter.begin());
//							else
//							{
//								for (int each_filter_element = 0; each_filter_element < filter_element_number_sum; each_filter_element++)
//								{
//									data_loc = filter_wise*filter_element_number_sum + each_filter_element;
//									target_proto->add_data(data_vec[data_loc]);
//
//								}
//							}
//						}
//						else
//						{
//							for (int each_filter_element = 0; each_filter_element < filter_element_number_sum; each_filter_element++)
//							{
//								data_loc = filter_wise*filter_element_number_sum + each_filter_element;
//								target_proto->add_data(data_vec[data_loc]);
//
//							}
//						}
//
//					}
//				}
//				else//prune bias correspondingly
//				{
//
//					target_proto->mutable_shape()->set_dim(0, blob_shape[0] - pruned_filter_num);
//					std::cout << target_proto->mutable_shape()->dim(0);
//
//					for (int bias_start = 0; bias_start < blob_shape[0]; bias_start++)
//					{
//						if (prune_bias_param.size() > 0)
//						{
//							if (bias_start == prune_bias_param[0])
//								prune_bias_param.erase(prune_bias_param.begin());
//							else
//								target_proto->add_data(data_vec[bias_start]);
//						}
//						else
//							target_proto->add_data(data_vec[bias_start]);
//
//					}
//				}
//			}
//			else if (source_layer_param.name() == former_conv_to_latter_conv[target_pruned_layer_name])//if the smallest filter in conv1_1 pruned, then continue prune channel correspondingly in conv1_2
//			{
//				if (blob == 0)//channel pruning
//				{
//					vector<int> temp_pruned_channel = prune_next_layer_channel_param;
//					int pruned_channel_num = pruned_filter_num;
//					target_proto->mutable_shape()->set_dim(1, blob_shape[1] - pruned_channel_num);
//					std::cout << target_proto->mutable_shape()->dim(0) << " " << target_proto->mutable_shape()->dim(1) << " " << target_proto->mutable_shape()->dim(2) << " " << target_proto->mutable_shape()->dim(3);
//
//					//int filter_element_number_sum = blob_shape[1] * blob_shape[2] * blob_shape[3];
//					int spatial_element_number_sum = blob_shape[2] * blob_shape[3];
//					int channel_num = blob_shape[1];
//					int data_loc = 0;
//					for (int filter_wise = 0; filter_wise < blob_shape[0]; filter_wise++)//for each filter
//					{
//						for (int channel_wise = 0; channel_wise < blob_shape[1]; channel_wise++)// for each channel
//						{
//							if (temp_pruned_channel.size()>0)
//							{
//								if (channel_wise == temp_pruned_channel[0])
//									temp_pruned_channel.erase(temp_pruned_channel.begin());
//								else
//								{
//									for (int element_wise = 0; element_wise < spatial_element_number_sum; element_wise++)
//									{
//										data_loc = (filter_wise*channel_num + channel_wise)*spatial_element_number_sum + element_wise;
//										target_proto->add_data(data_vec[data_loc]);
//									}
//								}
//							}
//							else
//							{
//								for (int element_wise = 0; element_wise < spatial_element_number_sum; element_wise++)
//								{
//									data_loc = (filter_wise*channel_num + channel_wise)*spatial_element_number_sum + element_wise;
//									target_proto->add_data(data_vec[data_loc]);
//								}
//							}
//						}
//						temp_pruned_channel = prune_next_layer_channel_param;
//					}
//				}
//				else
//				{
//					for (int j = 0; j < source_blobs[blob]->count(); ++j)
//						target_proto->add_data(data_vec[j]);
//				}
//			}
//			else//normal layers
//			{
//				for (int j = 0; j < source_blobs[blob]->count(); ++j)
//					target_proto->add_data(data_vec[j]);
//			}
//		}
//	}
//	WriteProtoToBinaryFile(pruned_param, pruned_network_name);
//	system("pause");
//	return 1;
//}
//
//
