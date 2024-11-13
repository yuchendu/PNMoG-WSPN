% clear;
% clc;
%% normal and abnormal fundus images loading
path_ab_K = '..\EvaluationMethod\193\193\(436-422)RemVesVesRGB(193)changename';
[ImgTensor_ab_K,file_ab_K] = fileloading(path_ab_K,2,'*.bmp');
ImgTensor_ab = double(ImgTensor_ab_K);
num_ab = size(ImgTensor_ab,3);

path_nor = '..\imageNormal';
[ImgTensor_nor,~] = fileloading(path_nor,2,'*.bmp');
ImgTensor_nor = double(ImgTensor_nor);

ImgTensor = cat(3,ImgTensor_ab,ImgTensor_nor);

%% parameter setting and model calling
% parameters setting for MoGRPCA_inexact
p = 1.1;
alpha = 1.0;
beta = 4.7;
lambda= 0.8;
weight_w = 0.1;
mask_thresh = 1;
MoG_param.K = 3;
MoG_param.ifpatch = 1;
MoG_param.P_row = 2;
MoG_param.P_col = 2;

Algorithm_iterMax = 40;
residual_E_mu = 5e-3;

RPCA_Param.p = p;
RPCA_Param.alpha = alpha;
RPCA_Param.beta = beta;
RPCA_Param.lambda = lambda;
RPCA_Param.mask_thresh = mask_thresh;
RPCA_Param.num_abnormal = num_ab;
RPCA_Param.weight_w = weight_w;

RPCA_Param.Algorithm_iterMax = Algorithm_iterMax;
RPCA_Param.residual_E_mu = residual_E_mu;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[lr_model, mog_model, r] = Schatten_MoG(ImgTensor,RPCA_Param,MoG_param);
Output_A = lr_model.A(:,:,1:num_ab);
Output_E = lr_model.E(:,:,1:num_ab);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% evaluation
path_groundtruth = '..\groundtruth\193_org_num';
[groundtruthTensor,~] = fileloading(path_groundtruth_K,1,'*.bmp');
[auc,pr,FPR,SE,PPv] = AUC_MAP(groundtruthTensor,Output_E);
disp(['p__',num2str(p),...
    '__alpha_',num2str(alpha),...
    '__beta_',num2str(beta),...
    '__lambda_',num2str(lambda),...
    '__p-MoG_',num2str(MoG_param.ifpatch),...
    '__patchSize_',strcat(num2str(MoG_param.P_row),'x',num2str(MoG_param.P_col)),...
    '__maskThresh_',num2str(mask_thresh),...
    '__weight_w_',num2str(weight_w),...
    '__N.nor2ab_',num2str(size(ImgTensor,3)-range),'_',num2str(range),...
    '__subAbIdx_',num2str(kk),...
    '__GaussK_',num2str(MoG_param.K),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)]);
store = ['p_',num2str(p),...
    '__alpha_',num2str(alpha),...
    '__beta_',num2str(beta),...
    '__lambda_',num2str(lambda),...
    '__p-MoG_',num2str(MoG_param.ifpatch),...
    '__patchSize_',strcat(num2str(MoG_param.P_row),'x',num2str(MoG_param.P_col)),...
    '__maskThresh_',num2str(mask_thresh),...
    '__weight_w_',num2str(weight_w),...
    '__N.nor2ab_',num2str(size(ImgTensor,3)-range),'_',num2str(range),...
    '__subAbIdx_',num2str(kk),...
    '__GaussK_',num2str(MoG_param.K),...
    '__AUC__',num2str(auc),'__AP_',num2str(pr)];

%% model save
mog_model.kappa = [];
% save([store,'.mat'],'Output_A','Output_E','mog_model','FPR','SE','PPv');
% save([store,'.mat'],'FPR','SE','PPv');
