#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <stdio.h>      
#include <math.h>
#include <map>

struct dataset{
    std::vector<std::string> label_;
    std::vector<std::vector<int> > attributes_;

    dataset(){label_.clear(); attributes_.clear();}
    dataset(std::vector<std::string>& label,std::vector<std::vector<int> >& attributes):label_(label),attributes_(attributes){}
};


class Bayes{

public:
    Bayes();
    void exect(std::string train_dataset, std::string test_dataset);


private:
    void read_file(std::string filename, struct dataset& data);
    void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c);
    void read_train_dataset(std::string filename);
    void read_test_dataset(std::string filename);
    void train_classfier();
    void calculate_attr_pro ();
    void calculate_label_pro();
    void test(struct dataset& test_data,std::string type);
    float calculte_nor_pro(int eachattributes, float variance, float mean);
    void result_anylyze();
    int  attribute_no_;

private:
  struct dataset train_dataset_;
  struct dataset test_dataset_;
  std::vector<std::map<int,int>> p_frequency_;
  std::vector<std::map<int,int>> n_frequency_;
  int countP_;
  int countN_;
  float  P_pone_;
  float  P_none_;
  std::vector<std::string> train_prediction_;
  std::vector<std::string> test_prediction_;

  

  


};

int main(int argc, char *argv[]){
    
    if(argc!=3){
        std::cout<<"wrong arguments"<<std::endl;
        return 0;
    }
   
    std::string trainFilename(argv[1]);
    std::string testFilename(argv[2]);

    Bayes bayes;
    bayes.exect(trainFilename,testFilename);
}

Bayes::Bayes(){
    attribute_no_=0;
}

void Bayes::read_file(std::string filename, struct dataset& data){
    std::ifstream in(filename);
    if(!in.is_open()){
        std::cout<<filename<<" cannot be opened"<<std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::vector<std::vector<std::pair<int,int>>>  result;
    
    while(getline(in,line)){
        //std::cout<<line<<std::endl;
        std::vector<std::string> v;
        if(line!=""){
            SplitString(line,v," ");
            data.label_.push_back(v[0]);
            std::vector<std::pair<int,int>> single_element_attri;
            for(size_t i=1;i<v.size();i++){
                std::vector<std::string> tokens;
                SplitString(v[i],tokens,":");
                single_element_attri.push_back(std::make_pair(stoi(tokens[0]),stoi(tokens[1])));
                if(stoi(tokens[0])>attribute_no_)
                    attribute_no_=stoi(tokens[0]);
            }
            result.push_back(single_element_attri);
        
        }


        
    }
    for(size_t i=0;i<result.size();i++){
            std::vector<int> attribute_temp(attribute_no_,0);
            for(auto atr_pair: result[i])
                attribute_temp[atr_pair.first-1]=atr_pair.second;
            
            data.attributes_.push_back(attribute_temp);
        }

    in.close();

}

void Bayes::SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        
        v.push_back(s.substr(pos1, pos2-pos1));
        
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void Bayes::read_train_dataset(std::string filename){
    read_file(filename,train_dataset_);
}

void Bayes::read_test_dataset(std::string filename){
    read_file(filename,test_dataset_);
}

void Bayes::calculate_attr_pro(){

    for(size_t i=0;i<train_dataset_.attributes_[0].size();i++){

        std::map<int,int> pos_map;
        std::map<int,int> neg_map;
        std::map<int,int>::iterator it;
        for(size_t j=0; j<train_dataset_.attributes_.size();j++){
            if(train_dataset_.label_[j]=="+1"){
                //std::cout<<"pos:"<<std::endl;
                it=pos_map.find(train_dataset_.attributes_[j][i]);
                if(it!=pos_map.end()){
                    it->second++;
                }
                else{
                   pos_map.emplace(train_dataset_.attributes_[j][i],1);
                }
                

            }
            else{
                //std::cout<<"neg:"<<std::endl;
                it=neg_map.find(train_dataset_.attributes_[j][i]);
                if(it!=neg_map.end()){
                    it->second++;
                }
                else{
                    neg_map.emplace(train_dataset_.attributes_[j][i],1);
                }
            }

        }
        p_frequency_.push_back(pos_map);
        n_frequency_.push_back(neg_map);


    }

    

     

}

void Bayes::calculate_label_pro(){
    countP_=0;
    countN_=0;
    for(int i=0;i<(int)train_dataset_.label_.size();i++){
        if(train_dataset_.label_[i]=="+1")
            countP_++;
        else 
            countN_++;
    }
    P_pone_=(float)countP_/(float)train_dataset_.label_.size();
    P_none_=(float)countN_/(float)train_dataset_.label_.size();
}

void Bayes::train_classfier(){
    calculate_attr_pro();
    calculate_label_pro();
}

void Bayes::test(struct dataset& test_data,std::string type){

    for(int i=0;i<(int)test_data.attributes_.size();i++){
        
        float P_p=log10(P_pone_);
        float P_n=log10(P_none_);

        for(size_t j=0;j<test_data.attributes_[0].size();j++){
            std::map<int,int>::const_iterator it;
            it=p_frequency_[j].find(test_data.attributes_[i][j]);
            int pos_fre;
            if(it!=p_frequency_[i].end()){
                pos_fre=it->second;
            }
            else{
                pos_fre=0;
            }
            it=n_frequency_[j].find(test_data.attributes_[i][j]);
            int neg_fre;
            if(it!=n_frequency_[i].end()){
                neg_fre=it->second;
            }
            else{
                neg_fre=0;
            }

            float predict_p=(float)(pos_fre)/(float)(countP_);
            //if(predict_p=)
            P_p+=log10(predict_p);
            float predict_n=(float)(neg_fre)/(float)(countN_);
            P_n+=log10(predict_n);
        }
        if(type=="test"){
            if(P_p>P_n){
               test_prediction_.push_back("+1");
            }
            else{
                test_prediction_.push_back("-1");
            }
        }
        
        if(type=="train"){
            if(P_p>P_n){
               train_prediction_.push_back("+1");
            }
            else{
               train_prediction_.push_back("-1");
            }
        }




    }
}



float Bayes::calculte_nor_pro(int eachattributes, float variance, float mean){

    float temp1=(eachattributes-mean)*(eachattributes-mean)*(-1);
    float temp2=2*variance;
    float temp3=(float)sqrt(2*M_PI*variance);
    float normal=(exp(temp1/temp2))/temp3;
    return normal;
}

void Bayes::exect(std::string train_dataset, std::string test_dataset){
    read_train_dataset(train_dataset);
    read_test_dataset(test_dataset);

    train_classfier();


    test(train_dataset_,"train");
    test(test_dataset_,"test");

    result_anylyze();
}

void Bayes::result_anylyze(){
    
    int TP=0,FN=0,FP=0,TN=0;
    
    for(int i=0;i<(int)train_dataset_.label_.size();i++){
        if(train_dataset_.label_[i]=="+1" && train_prediction_[i]=="+1")
            TP++;
        else if(train_dataset_.label_[i]=="+1" && train_prediction_[i]=="-1")
            FN++;
        else if(train_dataset_.label_[i]=="-1" && train_prediction_[i]=="+1")
            FP++;
        else if(train_dataset_.label_[i]=="-1" && train_prediction_[i]=="-1")
            TN++; 
    }

    std::cout<<TP<<" "<<FN<<" "<<FP<<" "<<TN<<std::endl;

    TP=FN=FP=TN=0;

    for(int i=0;i<(int)test_dataset_.label_.size();i++){
        if(test_dataset_.label_[i]=="+1" && test_prediction_[i]=="+1")
            TP++;
        else if(test_dataset_.label_[i]=="+1" && test_prediction_[i]=="-1")
            FN++;
        else if(test_dataset_.label_[i]=="-1" && test_prediction_[i]=="+1")
            FP++;
        else if(test_dataset_.label_[i]=="-1" && test_prediction_[i]=="-1")
            TN++; 
    }
    std::cout<<TP<<" "<<FN<<" "<<FP<<" "<<TN<<std::endl;
}