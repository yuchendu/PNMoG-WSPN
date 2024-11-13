function [AUC,map,FPR,SE,PPv] = AUC_MAP(groundtruth,inputTensor)

Mat_Truth = double(tenmat(groundtruth,3)');
Mat_Truth(Mat_Truth>0) = 1;
mask = imread('mask.bmp');
mask = mask(:,:,1);

[imgRows,imgCols,~] = size(inputTensor);
E = inputTensor;
E_posi = E;
E_posi = E_posi/max(abs(E_posi(:)))*255;
E_posi(E_posi<0) = 0;

E_nega = E;
E_nega = E_nega/max(abs(E_nega(:)))*255;
E_nega(E_nega>0) = 0;

E = E_posi + (-E_nega);
E = double(tenmat(E,3)');

[imgDims, imgNums] = size(Mat_Truth);
residualImg = E .* repmat(mask(:), 1, imgNums) ;
residualImg = residualImg/255;

groundtruthImg = Mat_Truth;

%dilate groundtruthImg
groundtruthDilateImg = zeros(imgDims, imgNums);
se=strel('square',5');
for i = 1:imgNums
    img = reshape(Mat_Truth(:,i), imgRows, imgCols);
    dimg = imdilate(img, se);
    groundtruthDilateImg(:,i) = dimg(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SE = [];
    PPv = SE;FPR=SE;ACC=SE;NPv=SE;

    rgs = 0:1/255:1;
    indx = 1;
    for i = rgs
        residualBwImg = im2bw(residualImg, i);
        residualDilateImg = zeros(imgDims, imgNums);
        for j = 1:imgNums
            img = reshape(residualBwImg(:,j), imgRows, imgCols);
            dimg = imdilate(img, se);
            %         figure(j);
            %         imshow(dimg);
            residualDilateImg(:,j) = dimg(:);
        end

        %calculate TP, FN, FP, TN at i
        intersectionImg = groundtruthImg & residualBwImg;
        TP = sum(intersectionImg(:));

        intersectionImg = groundtruthImg & residualDilateImg;
        groundtruthImgSum = sum(groundtruthImg(:));
        FN = groundtruthImgSum-sum(intersectionImg(:));

        intersectionImg = groundtruthDilateImg & residualBwImg;
        residualBwImgSum = sum(residualBwImg(:));
        FP = residualBwImgSum-sum(intersectionImg(:));

        unionImg = groundtruthImg | residualBwImg;

        TN = imgDims*imgNums-TP-FN-FP; % the equation to calculate TN

        if TP+FP ~= 0  % it makes no sense when TP and FP simultaneously equal to zeros
            elemSE = TP/(TP+FN+1e-8);
            elemFPR = 1-TN/(TN+FP+1e-8);
            elemPPv =  TP/(TP+FP+1e-8);
            elemNPv = TN/(TN+FN+1e-8);
            elemACC = (TP+TN)/(imgDims*imgNums);

            PPv(indx) =elemPPv;
            SE(indx) = elemSE;
            FPR(indx) = elemFPR;
            ACC(indx) = elemACC;
            NPv(indx) = elemNPv;
            indx = indx+1;
        else
            break;
        end

    end
   
 PPv=AddPPvElem(PPv');
    PPv=PPv';
    SE=AddTPRElem(SE');
    SE=SE';
    FPR=AddFPRElem(FPR');
    FPR=FPR';
    
    [fprSort, fprSortInd] = sort(FPR,'ascend');
    AUC = trapz([0 fprSort 1],[0 SE(fprSortInd) 1]);


    [seSort, seSortInd] = sort(SE,'ascend');
    map = trapz([0 seSort 1],[0 PPv(seSortInd) 1]);
%     fprintf('auc = %f, map = %f\n', AUC_bigben, map);

    % plot roc curve
    figure(1),
    plot(FPR,SE,'r-','linewidth',2);
    xlabel('FPR');
    ylabel('TPR');
    axis([0 1 0 1]);
    legend(['Our algorithm, AUC: ',num2str(AUC_bigben)]);
    title('TPR-FPR value');
    %grid on;

    figure(2),
    plot(SE,PPv,'k-','linewidth',2);
    xlabel('SE');
    ylabel('PPv');
    axis([0 1 0 1]);
    legend(['Our algorithm, MAP: ',num2str(map)]);
    title('Sensitivity-Predictitive value');
    %grid on;

end

function NewVal=AddFPRElem(OldElem)
    [M N]=size(OldElem);
    tmp=zeros(M+2,1);
    Sta=OldElem(1,1);
    End=OldElem(end,1);
    tmp(1,1)=1;
    tmp(M+2,1)=0;
    tmp(2:M+1,1)=OldElem;
    NewVal = tmp;
end

function NewVal=AddPPvElem(OldElem)
    [M N]=size(OldElem);
    tmp=zeros(M+2,1);
    Sta=OldElem(1,1);
    End=OldElem(end,1);
    tmp(1,1)=0;
    tmp(M+2,1)=1;
    tmp(2:M+1,1)=OldElem;
    NewVal = tmp;
end
function NewVal=AddTPRElem(OldElem)
    [M N]=size(OldElem);
    tmp=zeros(M+2,1);
    Sta=OldElem(1,1);
    End=OldElem(end,1);
    tmp(1,1)=1;
    tmp(M+2,1)=0;
    tmp(2:M+1,1)=OldElem;
    NewVal = tmp;
end
