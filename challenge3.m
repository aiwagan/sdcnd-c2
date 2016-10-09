%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% https://kyamagu.github.io/mexopencv/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;

f = 630.15; % focal distance for the two cameras
b = 0.12; % baseline, in m.

% main points left camera
CxLeft = 421;
CyLeft = 317;

% main points right camera
CxRight = 399;
CyRight = 317;
                
% I get the list of left and right images
ILeftList = dir('sub\img_CAMERA1_*left.jpg');
IRightList = dir('sub\img_CAMERA1_*right.jpg');

OldImLeft = [];

position = [0;0;0];
vecPosition = position';
vecPosition = repmat(vecPosition, 5, 1);

figure;
plot3(position(1),position(2),position(3));
view([0 0]);

lenMatch = [];
lenIn = [];

for index = 1:size(ILeftList,1)
    ImRight = rgb2gray(imread(sprintf('sub/%s',IRightList(index).name)));
    ImLeft = rgb2gray(imread(sprintf('sub/%s',ILeftList(index).name)));

    ptsLeft  = detectSURFFeatures(ImLeft, 'MetricThreshold', 1, ...
        'ROI', [1 3*size(ImLeft,1)/4+1 size(ImLeft,2) size(ImLeft,1)/4]);
    ptsRight = detectSURFFeatures(ImRight, 'MetricThreshold', 1, ...
        'ROI', [1 3*size(ImLeft,1)/4+1 size(ImLeft,2) size(ImLeft,1)/4]);

    [featuresLeft,  validPtsLeft]  = extractFeatures(ImLeft,  ptsLeft);
    [featuresRight, validPtsRight] = extractFeatures(ImRight, ptsRight);

    indexPairs = matchFeatures(featuresLeft, featuresRight);

    matchedLeft  = validPtsLeft(indexPairs(:,1));
    matchedRight = validPtsRight(indexPairs(:,2));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    PLeft = [f 0 CxLeft 0; 0 f CyLeft 0; 0 0 1 0];

    ApLeft = [matchedLeft.Location'; ones(1, size(matchedLeft.Location, 1))];

    ALeft = PLeft \ ApLeft;

    ALeft = ALeft(1:3, :);

    disparite = matchedLeft.Location - matchedRight.Location;

    disparite = disparite(:,1)';

    Az = f * b ./ (disparite);

    Az = repmat(Az,3,1);

    A = ALeft .* Az;

    matchedFeaturesLeft = featuresLeft(indexPairs(:,1), :);
    
    if ~isempty(OldImLeft)

        indexPairs = matchFeatures(matchedFeaturesLeft, OldMatchedFeaturesLeft);
        
        newA = A(:, indexPairs(:,1));
        newOldAB = oldA(:, indexPairs(:,2));
        
        NewMatchedLeft    = matchedLeft(indexPairs(:,1));
        NewOldMatchedLeft = OldMatchedLeft(indexPairs(:,2));
        
        camMatrix = [f 0 CxLeft; 0 f CyLeft; 0 0 1];
                
        [rvec, tvec, inliers] = cv.solvePnPRansac(double(newOldAB)', ...
            double(NewMatchedLeft.Location), double(camMatrix));
        rot = cv.Rodrigues(rvec);
        
        lenMatch = [lenMatch; length(indexPairs)];
        lenIn = [lenIn; length(inliers)];
                
        % [R T] = PointTransform3D(newOldAB, newA);

        position = rot' * position - tvec;
        
        % position = (R * position) + T;
        
        vecPosition = [vecPosition; (sum(vecPosition(end-3:end,:)) + 2*position') / 6];
        
        subplot(1,2,1);
        plot3(vecPosition(:,1),vecPosition(:,2),vecPosition(:,3));
        view([0 0]);
        drawnow;
        hold on;
        
        subplot(1,2,2);
        showMatchedFeatures(ImLeft,OldImLeft,NewMatchedLeft,NewOldMatchedLeft)
        hold on;
        %text(double(NewMatchedLeft.Location(:,1)),double(NewMatchedLeft.Location(:,2)),num2cell(newA(3,:)),'Color','blue')
        drawnow;
    end
    
    OldImLeft = ImLeft;
    OldMatchedFeaturesLeft = matchedFeaturesLeft;
    OldMatchedLeft = matchedLeft;
    
    oldA = A;
end

figure;
plot(lenMatch);

figure;
plot(lenIn);
