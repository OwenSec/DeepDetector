function Test_ImageNet_Deepfool_CaffeNet()
caffe.set_mode_gpu();
net_weights = 'bvlc_reference_caffenet.caffemodel'; % CaffeNet's weights
ori_net_model = 'deploy_original.prototxt'; % CaffeNet's architecture
mod_net_model= 'deploy_removeSoftmax.prototxt';
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
global ori_net;
ori_net = caffe.Net(ori_net_model, net_weights, phase);
mod_net = caffe.Net(mod_net_model, net_weights, phase);

IMAGE_DIM = 227;

rootPath = ['/home/ll/DeepDetector/TestImagenet/Zebra/';
    '/home/ll/DeepDetector/TestImagenet/Panda/';
    '/home/ll/DeepDetector/TestImagenet/Cabbb/'];
correctLabel = [341, 389, 469];

for order=1:3
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
                oriEntropy = oneDEntropy(imageForEntropy,224,224);
                if (oriEntropy < 4)
                    xFinal = scalarQuantization(image,128);
                elseif (oriEntropy < 5)
                    xFinal = scalarQuantization(image,64);
                else
                    xFinal = imProcess(image,4,224,13);
                end
                oriProcessedLabel = ori_predicating(xFinal);
                oriFilteredPreTime = toc;
                
                tic;
                advEntropy = oneDEntropy(advForEntropy,227,227);
                if (advEntropy < 4)
                    advXFinal = scalarQuantization(advX,128);
                elseif (advEntropy < 5)
                    advXFinal = scalarQuantization(advX,64);
                else
                    advXFinal = imProcess(advX,4,224,13);
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
end

    function pLabel = ori_predicating(im)
        global ori_net;
        pLabel = ori_net.forward({im});
        pLabel = pLabel{1}';
        [~,pLabel] = max(pLabel);
    end

    function processedIm = normalization(im)
        processedIm = im;
        processedIm(processedIm < 0) = 0;
        processedIm(processedIm > 255) = 255;
    end

    function processedIm = crossMeanFilterOperations(im, head, tail, coefficient)
        processedIm = im;
        for row=head:tail
            for col=head:tail
                temp1 = im(row,col,1);
                temp2 = im(row,col,2);
                temp3 = im(row,col,3);
                for i=1:(head-1)
                    temp1 = temp1+im(row-i,col,1)+im(row+i,col,1)+im(row,col-i,1)+im(row,col+i,1);
                    temp2 = temp2+im(row-i,col,2)+im(row+i,col,2)+im(row,col-i,2)+im(row,col+i,2);
                    temp3 = temp3+im(row-i,col,3)+im(row+i,col,3)+im(row,col-i,3)+im(row,col+i,3);
                end
                processedIm(row,col,1) = temp1/coefficient;
                processedIm(row,col,2) = temp2/coefficient;
                processedIm(row,col,3) = temp3/coefficient;
            end
        end
    end

    function processedIm=scalarQuantization(im,interval)
        processedIm = fix(im/interval);
        processedIm = processedIm*interval;
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

    function processedIm=imProcess(im,head,tail,coefficient)
        im1 = scalarQuantization(im,43);
        im2 = crossMeanFilterOperations(im1,head,tail,coefficient);
        processedIm = generateFinalIm(im,im1,im2);
    end

    function entropy = oneDEntropy(inputImage,height,width)
    area = height*width;
    f1 = zeros(1,256);
    f2 = zeros(1,256);
    f3 = zeros(1,256);
    for i=1:height
        for j=1:width
            f1(inputImage(i,j,1)+1) = f1(inputImage(i,j,1)+1)+1;
            f2(inputImage(i,j,2)+1) = f2(inputImage(i,j,2)+1)+1;
            f3(inputImage(i,j,3)+1) = f3(inputImage(i,j,3)+1)+1;
        end
    end
    f1 = f1/area;
    f2 = f2/area;
    f3 = f3/area;
    H1 = 0;
    H2 = 0;
    H3 = 0;
    for i=1:256
        if f1(i) > 0
            H1 = H1 + f1(i)*log2(f1(i));
        end
        if f2(i) > 0
            H2 = H2 + f2(i)*log2(f2(i));
        end
        if f3(i) > 0
            H3 = H3 + f3(i)*log2(f3(i));
        end
    end
    entropy = -(H1+H2+H3)/3;
end
