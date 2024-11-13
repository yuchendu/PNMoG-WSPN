function [Im,files] = fileloading(path,layer,ImType,removelayer,num_name)
%load('num2name.mat');
if nargin < 1
    disp('No Input Parameter!')
    return;
end
if nargin < 2
    layer = 0;
    ImType = '*.bmp';
    removelayer = [];
    num_name = 1;
end
if nargin < 3
	ImType = '*.bmp';
    removelayer = [];
    num_name = 1;
end  
if nargin < 4
    removelayer = [];
    num_name = 1;
end
if nargin < 5
    num_name = 1;
end
if ~exist(path,'dir')
    error(disp('Invalid file path'));
end

files = dir(fullfile(path,ImType));
lengthFiles = length(files);
Im = [];
layerIndex = 1;
for index = 1:lengthFiles
    if strcmp(files(index).name,'.') || strcmp(files(index).name,'..')
        continue;
    end
    filepath = strcat(path,'\',files(index).name);
    tempIm = imread(filepath);
    if layer>0 && layer<4
        enroll = 1;
        for kk = 1:length(removelayer)
            if strcmp(num2name_193(removelayer(kk),num_name),files(index).name)
                enroll = 0;
                break;
            end
        end
                
        if ~enroll
            continue;
        else
            ImLayer = tempIm(:,:,layer);
        end
    elseif layer == 0
        ImLayer = rgb2gray(tempIm);
    else
        return;
    end
    Im(:,:,layerIndex) = ImLayer;
    layerIndex = layerIndex + 1;
end


end