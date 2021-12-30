faces = load('faces.mat');
raw_images = faces.raw_images;
bigMatrix = zeros(86, 3000);


for i = 1:length(raw_images)
    X = raw_images(i);
    faceMat = reshape(X{1, 1}, [1,3000]);
    bigMatrix(i,:) = faceMat;
    
end


[coeff,score,latent,tsquared,explained,mu] = pca(bigMatrix);
cumVar = cumsum(latent);
x = (1:85);
y = cumVar;
plot(x,y)
xlabel('Principal Component')
ylabel('Cummulative Variance')
title('Cummulative Variance over all principal components');
figure;
normCoefs = zeros(3000, 10);
for i = 1:10
    subplot(2,5,i)
    normCoef1 = (coeff(:,i) - min(coeff(:,i)))/(max(coeff(:,i)) - min(coeff(:,i)));
    eigFace = reshape(normCoef1, [60,50]);
    normCoefs(:,i) = normCoef1;

    img=eigFace;  

    imshow(img);
    
end

function X = facial_recognition(input)
    faces = load('faces.mat');
    raw_images = faces.raw_images;
    testImage = imread(input);
    tempImage = imresize(rgb2gray(testImage), [60,50]);
    testMat = reshape(tempImage, [3000,1]);
    Euclideandistance=[];
    bigMatrix = zeros(86, 3000);


    for i = 1:length(raw_images)
        X = raw_images(i);
        faceMat = reshape(X{1, 1}, [1,3000]);
        bigMatrix(i,:) = faceMat;

    end
    
    [coeff] = pca(bigMatrix);

    bigW = bigMatrix*coeff;
    testW = double(testMat)'*coeff;

    for i=1:86
        temp=bigW(i,:)-testW;
        Euclideandistance(i,:)= temp;
    end

    tem=[];
    for i=1:size(Euclideandistance,1)
        k=Euclideandistance(i,:);
        tem(i)=sqrt(sum(k.^2));
    end

    [MinEuclid, index]=min(tem);
    if(MinEuclid<3600)

        outputimage=(raw_images{:,index});
        figure;
        subplot(1,2,1), imshow(input), title('input face')
        subplot(1,2,2), imshow(outputimage), title('recognised face: access granted')
       
        
    else
        disp('No matches found');
        disp('You are not allowed to enter this system');
        figure, imshow(input), title('input face: access denied')
        outputimage=0;
    end
end

