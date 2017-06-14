function detecting_ImageNet_examples_Crafted_By_DeepFool_CaffeNet()
caffe.set_mode_cpu();
net_weights = '~/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'; % CaffeNet's weights
ori_net_model = '~/caffe/models/bvlc_reference_caffenet/deploy_original.prototxt'; % GoogLenet's architecture
mod_net_model= '~/caffe/models/bvlc_reference_caffenet/deploy_removeSoftmax.prototxt';
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
ori_net = caffe.Net(ori_net_model, net_weights, phase);
mod_net = caffe.Net(mod_net_model, net_weights, phase);

IMAGE_DIM = 227;

rootPath = ['~/n02391049_ps/';
    '~/n02510455_ps/';
    '~/n07753275_ps/';
    '~/n02930766_ps/'];
correctLabel = [341, 389, 954, 469];

for order=1:4
    imageRootPath = rootPath(order,:);
    imageList = dir(fullfile(imageRootPath));
    imageCorrectLabel = correctLabel(order);
    imageNumber = size(imageList,1) ; %contains directory '.' and '..'
    
    wrongAlreadyNumber = 0;
    failedNumber = 0;
    testOriNumber = 0;
    TP = 0;
    FN = 0;
    FP = 0;
    TTP=0;
    
    oriPreTimeSum = 0;
    advPreTimeSum = 0;
    advGenTimeSum = 0;
    oriFilteredPreTimeSum = 0;
    advFilteredPreTimeSum = 0;
    
    for k=3:imageNumber
        image = imread([imageRootPath, imageList(k).name]);
        s =size(image);
        [~, col] = size(s);
        if (col ~= 3)
            continue;
        end
        if (s(1) ~= 224 || s(2) ~=224 || s(3) ~= 3)
            continue;
        end
        
        oriEntropy = 0;
        advEntropy = 0;
        
        imageForEntropy = image;
        
        image = permute(image, [2, 1, 3]);  % flip width and height
        image = image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
        image = single(image);  % convert from uint8 to single
        image = imresize(image, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
        tic;
        oriLabel = ori_predicating(image);
        oriPreTime = toc;
        
        if (oriLabel ~= imageCorrectLabel)
            wrongAlreadyNumber = wrongAlreadyNumber + 1;
            continue;
        end
        
        if (oriLabel == imageCorrectLabel)
            tic;
            [perturbations, ~, ~, ~] = adversarial_DeepFool_caffe(image,mod_net);
            advX = image + perturbations;
            advX = normalization(advX);
            advGenTime = toc;
            
            tic;
            advLabel = ori_predicating(advX);
            advPreTime = toc;
            
            if (advLabel == oriLabel)
                failedNumber = failedNumber + 1;
                continue;
            end
            
            if (advLabel ~= oriLabel)
                testOriNumber = testOriNumber + 1;
                
                fprintf('%d: %s\n', testOriNumber, [imageRootPath, imageList(k).name]);
                
                advForEntropy = uint8(advX);
                advForEntropy = advForEntropy(:, :, [3, 2, 1]);
                advForEntropy = permute(advForEntropy, [2, 1, 3]);
                
                tic;
                oriEntropy = image2DEntropy55(imageForEntropy, 224);
                if (oriEntropy < 8.5)
                    xFinal = twoScalarQuantization(image);
                elseif (oriEntropy < 9.5)
                    xFinal = fourScalarQuantization(image);
                else
                    xAfterScalarQuantization = sixScalarQuantization(image);
                    xAfterMeanFilter = fiveCrossMeanFilter(xAfterScalarQuantization);
                    xFinal = generateFinalIm(image,xAfterScalarQuantization,xAfterMeanFilter);
                end
                oriProcessedLabel = ori_predicating(xFinal);
                oriFilteredPreTime = toc;
                
                tic;
                advEntropy = image2DEntropy55(advForEntropy,227);
                if (advEntropy < 8.5)
                    advXFinal = twoScalarQuantization(image);
                elseif (advEntropy < 9.5)
                    advXFinal = fourScalarQuantization(image);
                else
                    advXAfterScalarQuantization = sixScalarQuantization(advX);
                    advXAfterMeanFilter = fiveCrossMeanFilter(advXAfterScalarQuantization);
                    advXFinal = generateFinalIm(advX,advXAfterScalarQuantization, advXAfterMeanFilter);
                end
                advProcessedLabel = ori_predicating(advXFinal);
                advFilteredPreTime = toc;
                
                if (advLabel ~= advProcessedLabel)
                    TP = TP + 1;
                    if (advProcessedLabel == oriLabel)
                        TTP = TTP + 1;
                    end
                end
                if (advLabel == advProcessedLabel)
                    FN = FN + 1;
                end
                if (oriLabel ~= oriProcessedLabel)
                    FP = FP + 1;
                end
                oriPreTimeSum=oriPreTimeSum+oriPreTime;
                advPreTimeSum=advPreTimeSum+advPreTime;
                oriFilteredPreTimeSum=oriFilteredPreTimeSum+oriFilteredPreTime;
                advFilteredPreTimeSum=advFilteredPreTimeSum+advFilteredPreTime;
                advGenTimeSum=advGenTimeSum+advGenTime;
                fprintf('%f-%f\n', oriEntropy, advEntropy);
                fprintf('%d - %d - %d , TTP = %d, TP = %d, FN = %d, FP = %d\n', testOriNumber, wrongAlreadyNumber, failedNumber,TTP,TP,FN,FP);
                fprintf('%f--%f--%f--%f--%f\n', oriPreTime, advPreTime, oriFilteredPreTime, advFilteredPreTime, advGenTime);
                fprintf('%f--%f--%f--%f--%f\n', oriPreTimeSum, advPreTimeSum, oriFilteredPreTimeSum, advFilteredPreTimeSum, advGenTimeSum);
            end
        end
    end
    Recall = TP/(TP+FN);
    Precision = TP/(TP+FP);
    fprintf('************************************************************************************');
    fprintf('\nRecall = %f\n', Recall);
    fprintf('Precision = %f\n\n', Precision);
    fprintf('************************************************************************************\n');
end
caffe.reset_all(); % reset caffe

    function pLabel = ori_predicating(im)
        pLabel = ori_net.forward({im});
        pLabel = pLabel{1}';
        [~,pLabel] = max(pLabel);
    end


    function pLabel = mod_predicating(im)
        pLabel = mod_net.forward({im});
        pLabel = pLabel{1}';
        [~,pLabel] = max(pLabel);
    end

    function processedIm = normalization(im)
        processedIm = im;
        processedIm(processedIm < 0) = 0;
        processedIm(processedIm > 255) = 255;
    end

    function processedIm = twoScalarQuantization(im)
        processedIm = im;
        processedIm(processedIm < 128) = 0;
        processedIm(processedIm > 127) = 128;
    end

    function processedIm = fourScalarQuantization(im)
        processedIm = im;
        processedIm(processedIm < 64) = 0;
        processedIm(processedIm < 128 & processedIm > 63) = 64;
        processedIm(processedIm < 192 & processedIm > 127) = 128;
        processedIm(processedIm > 191) = 192;
    end

    function processedIm = sixScalarQuantization(im)
        processedIm = im;
        processedIm(processedIm < 50) = 0;
        processedIm(processedIm < 100 & processedIm > 49) = 50;
        processedIm(processedIm < 150 & processedIm > 99) = 100;
        processedIm(processedIm < 200 & processedIm > 149) = 150;
        processedIm(processedIm < 250 & processedIm > 199) = 200;
        processedIm(processedIm > 249) = 250;
    end

    function processedIm = fiveCrossMeanFilter(im)
        processedIm = im;
        for c=1:3
            for w=3:225
                for h=3:225
                    processedIm(w,h,c) = (im(w,h,c)+im(w-1,h,c)+im(w,h+1,c)+im(w+1,h,c)+im(w,h-1,c)+im(w-2,h,c)+im(w,h+2,c)+im(w+2,h,c)+im(w,h-2,c))/9;
                    processedIm(w,h,c) = uint8(processedIm(w,h,c));
                end
            end
        end
    end

    function processedIm = generateFinalIm(im,im1,im2)
        processedIm = im;
        for c=1:3
            for w=1:227
                for h=1:227
                    if (abs(im(w,h,c) - im1(w,h,c)) < abs(im(w,h,c) - im2(w,h,c)))
                        processedIm(w,h,c) = im1(w,h,c);
                    else
                        processedIm(w,h,c) = im2(w,h,c);
                    end
                end
            end
        end
    end

    function processedIm = meanFilter(im,size)
        im = single(im);
        processedIm = im;
        for c=1:3
            for w=3:(size-2)
                for h=3:(size-2)
                    processedIm(w,h,c)=(im(w-2,h-2,c)+im(w-1,h-2,c)+im(w,h-2,c)+im(w+1,h-2,c)+im(w+2,h-2,c)+...
                        im(w-2,h-1,c)+im(w-1,h-1,c)+im(w,h-1,c)+im(w+1,h-1,c)+im(w+2,h-1,c)+...
                        im(w-2,h,c)+im(w-1,h,c)+im(w,h,c)+im(w+1,h,c)+im(w+2,h,c)+...
                        im(w-2,h+1,c)+im(w-1,h+1,c)+im(w,h+1,c)+im(w+1,h+1,c)+im(w+2,h+1,c)+...
                        im(w-2,h+2,c)+im(w-1,h+2,c)+im(w,h+2,c)+im(w+1,h+2,c)+im(w+2,h+2,c))/25;
                end
            end
        end
    end

    function entropy = image2DEntropy55(im,size)
        im2 = meanFilter(im,size);
        R_entropy = single(0);
        G_entropy = single(0);
        B_entropy = single(0);
        R_num = zeros(256,256);
        G_num = zeros(256,256);
        B_num = zeros(256,256);
        pmf_R = single(zeros(256,256));
        pmf_G = single(zeros(256,256));
        pmf_B = single(zeros(256,256));
        for w=1:size
            for h=1:size
                R_val=uint8(im(w,h,1));
                G_val=uint8(im(w,h,2));
                B_val=uint8(im(w,h,3));
                R_avg=uint8(im2(w,h,1));
                G_avg=uint8(im2(w,h,2));
                B_avg=uint8(im2(w,h,3));
                R_num(R_val+1,R_avg+1)=R_num(R_val+1,R_avg+1)+1;
                G_num(G_val+1,G_avg+1)=G_num(G_val+1,G_avg+1)+1;
                B_num(B_val+1,B_avg+1)=B_num(B_val+1,B_avg+1)+1;
            end
        end
        for i=1:256
            for j=1:256
                pmf_R(i,j)=single(R_num(i,j))/(size*size);
                pmf_G(i,j)=single(G_num(i,j))/(size*size);
                pmf_B(i,j)=single(B_num(i,j))/(size*size);
            end
        end
        for i=1:256
            for j=1:256
                if (pmf_R(i,j) ~= 0)
                    R_entropy = R_entropy + pmf_R(i,j)*log10(pmf_R(i,j))/log10(2);
                end
                if (pmf_G(i,j) ~= 0)
                    G_entropy = G_entropy + pmf_G(i,j)*log10(pmf_G(i,j))/log10(2);
                end
                if (pmf_B(i,j) ~= 0)
                    B_entropy = B_entropy + pmf_B(i,j)*log10(pmf_B(i,j))/log10(2);
                end
            end
        end
        R_entropy=-R_entropy;
        G_entropy=-G_entropy;
        B_entropy=-B_entropy;
        entropy = (R_entropy+G_entropy+B_entropy)/3;
    end
end
