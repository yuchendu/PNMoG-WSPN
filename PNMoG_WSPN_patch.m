function [RPCA_model,MoG_model,residual]=PNMoG_WSPN_patch(X,RPCA_param,MoG_param)

if nargin < 1
    disp('input parameter required');
    return;
end

% transfer uint8 D to double D
X = double(X);

% parameter initialization
sz_X = size(X);
if nargin == 1
    RPCA_param.alpha = 1*0.8*sqrt(sz_X(1)*sz_X(2));
    RPCA_param.beta = 1;
    MoG_param.K = 3;
elseif nargin == 2
    RPCA_param.alpha = RPCA_param.alpha*sqrt(sz_X(1)*sz_X(2));
    MoG_param.K = 3;
elseif nargin == 3
    RPCA_param.alpha = RPCA_param.alpha*sqrt(sz_X(1)*sz_X(2));
end

p = RPCA_param.p;
alpha = RPCA_param.alpha;
beta = RPCA_param.beta;
lambda = RPCA_param.lambda;
mask_thresh = RPCA_param.mask_thresh;
num_abnormal = RPCA_param.num_abnormal;
weight_w = RPCA_param.weight_w;

P_row = MoG_param.P_row;
P_col = MoG_param.P_col;

% ALM initialization
D = double(tenmat(X,3)');
Y = D;% Y is the lagrangian operator
[U_init,S_init,V_init] = svd(Y,'econ');

w0 = (S_init(1,1)./diag(S_init)).^(weight_w);

SingularMax = S_init(1,1);
norm_inf = norm( Y(:), inf) / (beta/alpha);
dual_norm = max(SingularMax, norm_inf);
Y = Y / dual_norm;

A = U_init(:,1)*S_init(1,1)*V_init(:,1)';% A is the low rank matrix
E = D - A;% E is the sparse matrix

nv_init = 7/SingularMax; % this one can be tuned, nv is the augment lagrangian operator
rho = 1.5;          % this one can be tuned

dnorm = norm(D, 'fro');
tolProj = 1e-6 * dnorm;

% transfer matrix to tensor
Y = matten(Y',sz_X,3);
A = matten(A',sz_X,3);
E = matten(E',sz_X,3);

% MoG initialization
Mask_out = imread('mask_out.bmp');
Mask_disc = imread('mask.bmp');
rangeE = max(E(:))-min(E(:));
threshE = rangeE*0.01;
threshE = mask_thresh;
Mask2 = abs(E)>=threshE;
Mask_E_disc = and(Mask2, Mask_disc);

% load('mog_prior.mat');
MoG_model = init_MoG(E(:,:,1:num_abnormal),Mask_E_disc(:,:,1:num_abnormal),MoG_param);

% converge condition
Algorithm_iterMax = RPCA_param.Algorithm_iterMax;
residual_E_mu = RPCA_param.residual_E_mu;
residual_G_mu = RPCA_param.residual_E_mu*1e3;

disp('*** Algorithm optimization started ***');                                                                                                                         
Algorithm_iter = 0;
Algorithm_converge = 0;
while ~Algorithm_converge
    tic;
    Algorithm_iter = Algorithm_iter+1;
    disp(['The ',num2str(Algorithm_iter),'th iteration has been starting']);
    % Gauss refreshing
    if lambda~=0
        disp(' * MoG optimization started *');
        MoG_model_last = MoG_model;

        % E step
        MoG_model = expectation(E(:,:,1:num_abnormal),MoG_model);
        % M step
        MoG_model = maximization(E(:,:,1:num_abnormal),Mask_E_disc(:,:,1:num_abnormal),MoG_model);
        
        str_pi = [];str_mu = [];str_sigma = [];
        for k = 1:MoG_model.K
            str_pi = strcat(str_pi,'       pi',num2str(k));
            str_mu = strcat(str_mu,'       mu',num2str(k));
            str_sigma = strcat(str_sigma,' sigma^2_',num2str(k));
        end
        displayIndex = 110;
        disp(str_pi);
        disp(MoG_model.pi_k{displayIndex});
        disp(str_mu);
        disp(MoG_model.mu_k{displayIndex});
        disp(str_sigma);
        disp_sigma = [];
        for k = 1:MoG_model.K
            disp_sigma = [disp_sigma,diag(MoG_model.sigma_2_k{displayIndex}(:,:,k))];
        end
        disp(disp_sigma);
    
    end
    
    % RPCA update
    disp(' * RPCA optimization started *');
    nv = min(nv_init, 1e6);
    % update A
    [U,S,V] = svd(double(tenmat(X-E+1/nv*Y,3)'),'econ');
    sigma = diag(S);
    w = w0.*alpha/nv;
    Delta = schattenThreshold(sigma,p,w);
    A_new = U*Delta*V';
    A_new = double(matten(A_new',sz_X,3));
    
    % update E
    if lambda~=0
    % calculate the coefficient matrix W, length of tensor has been
    % defined as W, so this parameter is renamed Q
    [H,W,M] = size(X(:,:,1:num_abnormal));
    K = MoG_model.K;
    Q1 = zeros(H,W,M,K);
    Q2 = zeros(H,W,M,K);
    Psi = zeros(2,2,K,num_abnormal);
    Mu = zeros(2,2,K,num_abnormal);
    Kappa = zeros(H,W,K,num_abnormal);
    for slice = 1:num_abnormal
        for k = 1:K
%             psi_k = reshape(diag(MoG_model.sigma_2_k{slice}(:,:,k)),[P_row,P_col]);
            psi_k = reshape(diag(MoG_model.sigma_2_k{slice}(:,:,k)),[2,2]);
            Psi(:,:,k,slice) = psi_k;
%             mu_k = reshape(MoG_model.mu_k{slice}(:,k),[P_row,P_col]);
            mu_k = reshape(MoG_model.mu_k{slice}(:,k),[2,2]);
            Mu(:,:,k,slice) = mu_k;
        end
        Kappa(:,:,:,slice) = MoG_model.kappa{slice};
    end
    Psi = permute(Psi,[1,2,4,3]);
    Mu = permute(Mu,[1,2,4,3]);
    Kappa = permute(Kappa,[1,2,4,3]);
    for k = 1:K
        for i = 1:2
            for j = 1:2
                Q1(i:H-2+i,j:W-2+j,:,k) = Q1(i:H-2+i,j:W-2+j,:,k)...
                    +bsxfun(@rdivide,Kappa(1:H-1,1:W-1,:,k),Psi(i,j,:,k));
                Q2(i:H-2+i,j:W-2+j,:,k) = Q2(i:H-2+i,j:W-2+j,:,k)...
                    +bsxfun(@times,bsxfun(@rdivide,Kappa(1:H-1,1:W-1,:,k),Psi(i,j,:,k)),Mu(i,j,:,k));
            end
        end
    end
    a_nume_1 = zeros(sz_X);
    a_demo_1 = zeros(sz_X);
    for k = 1:K
        a_nume_1(:,:,1:num_abnormal) = a_nume_1(:,:,1:num_abnormal) + Q2(:,:,:,k);
        a_demo_1(:,:,1:num_abnormal) = a_demo_1(:,:,1:num_abnormal) + Q1(:,:,:,k);
    end
    a_nume = lambda*a_nume_1 + nv*(X-A_new)+Y;
    a_demo = lambda*a_demo_1 + nv;
    a = a_nume./a_demo;
    zeta = beta./a_demo;
    clear a_nume_1 a_nume a_demo_1 a_demo;
    E_new = softThreshold(a,zeta);
    else
        a =  (X-A_new)+Y/nv;
        zeta = beta/nv;
        %clear a_nume_1 a_nume a_demo_1 a_demo;
        E_new = softThreshold(a,zeta);
    end
    
    % update Mask
    rangeE = max(E_new(:))-min(E_new(:));
    threshE = mask_thresh;
    Mask_new = (abs(E_new)>=threshE);
    Mask_E_disc = and(Mask_new,Mask_disc);
    
    Y = Y + nv*(X-A_new-E_new);
    nv_init = rho*nv_init;
    
    % stop Criterion 
    if lambda~=0
    mu_loss = norm(tensor(MoG_model.mu_k{1}-MoG_model_last.mu_k{1}))/norm(tensor(MoG_model_last.mu_k{1}));
    else
    mu_loss = 0;
    end
    error_recst = norm(tensor(X-A_new-E_new))/dnorm;
    error_converge = norm(tensor(A_new-A))/norm(tensor(A));
    
    residual = error_recst;
    
    t = toc;
    disp(['    reconstruction error: ',num2str(error_recst)]);
    disp(['    converge error: ',num2str(error_converge)]);
    disp(['    gauss error: ',num2str(mu_loss)]);
    disp(['  the ',num2str(Algorithm_iter),'th iteration finished, takes ',num2str(t)]);
    if error_recst<residual_E_mu...
            && error_converge<residual_E_mu...
            && mu_loss<residual_G_mu
        disp('Algorithm has converged');
        RPCA_model.A = A_new;
        RPCA_model.E = E_new;
        RPCA_model.U = U;
        RPCA_model.S = Delta;
        RPCA_model.V = V;
        Algorithm_converge = true;
    else
        if Algorithm_iter >= Algorithm_iterMax
            disp('Maximum iteration time has reached');
            RPCA_model.A = A_new;
            RPCA_model.E = E_new;
            RPCA_model.U = U;
            RPCA_model.S = Delta;
            RPCA_model.V = V;
        	break;
        end
        A = A_new;E = E_new;
    end
end
end


function MoG_model = init_MoG(X,M,MoG_model)% X is row matrix
% initialize the MoG model by the number of Gauss K, and if the Gauss unit
% is divided to patch format.
num_iid = size(X,3);

K = MoG_model.K;
P_row = MoG_model.P_row;
P_col = MoG_model.P_col;
L = 4;

%%%%%%%%%%%%%%%%%% patch desample to 2*2 %%%%%%%%%%%%%%%%%%
XX = zeros(size(X,1)+P_row-1,size(X,2)+P_col-1,size(X,3));
XX(ceil(P_row/2):end-floor(P_row/2),ceil(P_col/2):end-floor(P_col/2),:) = X;
X=XX;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate the valid mask
M_L = double(tenmat(M,3));% each row represents the validation of patch in THE image, corresponding to f_L

pi_k = cell(num_iid,1);
mu_k = cell(num_iid,1);
sigma_2_k = cell(num_iid,1);

for slice = 1:num_iid
    % patch tensor for each slice
    f_L = im2col(X(:,:,slice),[P_row,P_col],'sliding');
    % remove the masked patches
    f_L(:,M_L(slice,:)==0) = [];
    
    %%%%%%%%%% downsample %%%%%%%%%%%%
    range = ceil(P_row/2);
    step = floor(P_row/2);
    
    f_L_cube = matten(f_L',[P_row,P_col,size(f_L,2)],3);
    f_L_down = zeros(4,size(f_L,2));
    for jj = 1:2
        for ii = 1:2
            f_L_i = f_L_cube(1+step*(ii-1):range+step*(ii-1),1+step*(jj-1):range+step*(jj-1),:);
            f_L_i = double(tenmat(f_L_i,3)');
            f_L_down(ii+2*(jj-1),:) = mean(f_L_i);
        end
    end
    f_L = f_L_down;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    min_f_L = min(f_L,[],1);
    [~,max_pos] = max(min_f_L,[],2);
    max_m = f_L(:,max_pos(1));
    
    min_f_L = min(-f_L,[],1);
    [~,max_pos] = max(min_f_L,[],2);
    min_m = f_L(:,max_pos);    

    % set the K clustering centers
    m = (bsxfun(@times,linspace(0,K-1,K),min_m)+bsxfun(@times,linspace(K-1,0,K),max_m))/(K-1);
    
    % 
    [~,label] = max(bsxfun(@minus,m'*f_L,dot(m,m)'/2),[],1);
    [u,~,label] = unique(label);
    Np = size(f_L,2);% number of patches
    R = full(sparse(1:Np,label,1,Np,K,Np));% R:np*K
    
    % MoG parameter estimation
    uk_num = sum(R);
    pi_i = uk_num/Np;
    sigma_2_i = zeros(L,L,K);
    for k = 1:K
        uk_value = bsxfun(@times,f_L,R(:,k)');% the k th gauss's elements
        mu_i(:,k) = sum(uk_value,2)./uk_num(k);% average value of k th gauss
        temp_Sigma = bsxfun(@times,bsxfun(@minus,uk_value,mu_i(:,k)),R(:,k)');
        mat_Sigma = temp_Sigma*temp_Sigma'/uk_num(k);
        d = eig(mat_Sigma);
        if sum(d>1e-4,1)==size(mat_Sigma,1)
            sigma_2_i(:,:,k) = mat_Sigma;
        else
            sigma_2_i(:,:,k) = k*eye(size(mat_Sigma,1));
        end
    end
    pi_k{slice} = pi_i;
    mu_k{slice} = mu_i;
    sigma_2_k{slice} = sigma_2_i;
end
MoG_model.pi_k = pi_k;
MoG_model.mu_k = mu_k;
MoG_model.sigma_2_k = sigma_2_k;
% backup the initial parameters
MoG_model.pi_init = pi_k;
MoG_model.mu_init = mu_k;
MoG_model.sigma_2_init = sigma_2_k;
end


function MoG_model = expectation(X,MoG_model)
num_iid = size(X,3);

epsilon = 1e-2;
P_row = MoG_model.P_row;
P_col = MoG_model.P_col;

[H,W,~] = size(X);
    
%%%%%%%%%%%%%%%%%% patch desample to 2*2 %%%%%%%%%%%%%%%%%%
XX = zeros(size(X,1)+P_row-1,size(X,2)+P_col-1,size(X,3));
XX(ceil(P_row/2):end-floor(P_row/2),ceil(P_col/2):end-floor(P_col/2),:) = X;
X=XX;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kappa = cell(num_iid,1);
for slice = 1:num_iid
    % get patch matrix
    f_L = im2col(X(:,:,slice),[P_row,P_col],'sliding');
    
    %%%%%%%%%% downsample %%%%%%%%%%%%
    range = ceil(P_row/2);
    step = floor(P_row/2);
    
    f_L_cube = matten(f_L',[P_row,P_col,size(f_L,2)],3);
    f_L_down = zeros(4,size(f_L,2));
    for jj = 1:2
        for ii = 1:2
            f_L_i = f_L_cube(1+step*(ii-1):range+step*(ii-1),1+step*(jj-1):range+step*(jj-1),:);
            f_L_i = double(tenmat(f_L_i,3)');
            f_L_down(ii+2*(jj-1),:) = mean(f_L_i);
        end
    end
    f_L = f_L_down;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract the parameters of iid Gaussian
    pi_k = MoG_model.pi_k{slice};
    mu_k = MoG_model.mu_k{slice};
    sigma_2_k = MoG_model.sigma_2_k{slice};
    
    K = MoG_model.K;
    kappa_i = zeros(H,W,K);
    
    nume = zeros(H,W,K);
    for k = 1:K
        d = eig(sigma_2_k(:,:,k));
        y = pi_k(k)*mvnpdf(f_L',mu_k(:,k)',sigma_2_k(:,:,k));
        nume(:,:,k) = reshape(y,[H,W]);
    end
    nume(nume==0) = epsilon;
    demo = sum(nume,3);
    for k = 1:K
        kappa_i(:,:,k) = nume(:,:,k)./demo;
    end
    kappa{slice} = kappa_i;
end
MoG_model.kappa = kappa;
end


function MoG_model = maximization(X,M,MoG_model)
num_iid = size(X,3);
% get patch matrix
P_row = MoG_model.P_row;
P_col = MoG_model.P_col;
K = MoG_model.K;

%%%%%%%%%%%%%%%%%% set patch desample to 2*2 %%%%%%%%%%%%%%%%%%
XX = zeros(size(X,1)+P_row-1,size(X,2)+P_col-1,size(X,3));
XX(ceil(P_row/2):end-floor(P_row/2),ceil(P_col/2):end-floor(P_col/2),:) = X;
X=XX;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate the valid mask
M_L = double(tenmat(M,3));% each row represents the validation of patch in THE image, corresponding to f_L

mu_k = zeros(2*2,K);
sigma_2_k = zeros(2*2,2*2,K);
for slice = 1:num_iid
    % calculate the valid mask
    if sum(M_L(slice,:))==0
        MoG_model.pi_k{slice} = MoG_model.pi_init{slice};
        MoG_model.mu_k{slice} = MoG_model.mu_init{slice};
        MoG_model.sigma_2_k{slice} = MoG_model.sigma_2_init{slice};
        continue;
    end
    
    f_L = im2col(X(:,:,slice),[P_row,P_col],'sliding');
    f_L(:,M_L(slice,:)==0) = [];
    
    if isempty(f_L)
        continue;
    end
    
    %%%%%%%%%% downsample %%%%%%%%%%%%
    range = ceil(P_row/2);
    step = floor(P_row/2);
    
    f_L_cube = matten(f_L',[P_row,P_col,size(f_L,2)],3);
    f_L_down = zeros(4,size(f_L,2));
    for jj = 1:2
        for ii = 1:2
            f_L_i = f_L_cube(1+step*(ii-1):range+step*(ii-1),1+step*(jj-1):range+step*(jj-1),:);
            f_L_i = reshape(f_L_i,[],size(f_L_cube,3));
            f_L_down(ii+2*(jj-1),:) = mean(f_L_i);
        end
    end
    f_L = f_L_down;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load each patch's possibility
    kappa = MoG_model.kappa{slice};
    kappa = tensor(kappa);
    kappa = double(tenmat(kappa,3)');
    kappa(M_L(slice,:)==0,:) = [];

    MoG_model.pi_k{slice} = sum(kappa,1)/size(kappa,1);
    for k = 1:MoG_model.K
        % calculate mu
        mu_k(:,k) = sum(bsxfun(@times,kappa(:,k),f_L'),1)...
            /sum(kappa(:,k),1);
        % calculate sigma
        temp_Sigma = bsxfun(@minus,f_L,mu_k(:,k));
        temp_Sigma_kappa = bsxfun(@times,temp_Sigma,kappa(:,k)');
        mat_Sigma = temp_Sigma*temp_Sigma_kappa'./sum(kappa(:,k),1);
        sigma_2_k(:,:,k) = mat_Sigma;
    end
    MoG_model.mu_k{slice} = mu_k;
    MoG_model.sigma_2_k{slice} = sigma_2_k;

    MoG_model=MoG_check(MoG_model,slice);
end
end

function model = MoG_check(model,slice)
K = model.K;
pi_k = model.pi_k{slice};
sigma_2_k = model.sigma_2_k{slice};

if (~isempty(find(abs(sigma_2_k(:))<1e-4,1))) || (size(pi_k,2)~=K)
    model.pi_k{slice} = model.pi_init{slice};
    model.mu_k{slice} = model.mu_init{slice};
    model.sigma_2_k{slice} = model.sigma_2_init{slice};
end

for k = 1:K
    d = eig(sigma_2_k(:,:,k));
    if sum(d>1e-4,1)~=size(d,1)
        model.pi_k{slice} = model.pi_init{slice};
        model.mu_k{slice} = model.mu_init{slice};
        model.sigma_2_k{slice} = model.sigma_2_init{slice};
        return;
    end
end

end

function Delta = schattenThreshold(sigma,p,w)
% S_schattenThreshold[X] = argmin w*(X)^p + 1/2*(X-Y)^2
tau_p_w = (2.*w.*(1-p)).^(1/(2-p))+w.*p.*(2.*w.*(1-p)).^((p-1)/(2-p));
delta = abs(sigma);
for i = 1:20
    delta = abs(sigma)-w.*p.*(abs(delta)).^(p-1);
    if p==1
        break;
    end
end
Delta = sign(sigma).*delta;
Delta(abs(sigma)<abs(tau_p_w)) = 0;
Delta = diag(Delta);  
end

function [S] = softThreshold(Mat,mu)
% S_epsilon[X] = argmin mu*||X||_1 + 1/2*||X-Y||_fro2
% S_epsilon[X] = sgn(W) * max(|W|-mu,0)
MatThresh = abs(Mat)-mu;
MatThresh(MatThresh<0) = 0;
S = sign(Mat) .* MatThresh;
% rank
% diagS = diag(S);
% svp = length(find(diagS > 1/mu));
end
