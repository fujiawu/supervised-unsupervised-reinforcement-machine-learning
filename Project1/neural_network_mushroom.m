%% clean the workspace and command line
function neural_network_mushroom

clear;
clc;

%% read data
filename = 'mushroom\agaricus-lepiota.data';
fin = fopen(filename, 'r');
format = ['%s %s %s %s %s %s %s %s %s %s' ...
          '%s %s %s %s %s %s %s %s %s %s' ...
                       '%s %s %s'];
rawdata = textscan(fin,format,'delimiter',',');
fclose(fin);

%% shuffle the data
rawdata = shuffle(rawdata);

%% convert to normalized numeric
data = String2Matrix([rawdata(2:11), rawdata(13:23)]);
inputs = data(:,1:6)';
targets = String2BinaryMaxtrix(rawdata{1});


%% k-fold partition
cv = cvpartition(length(targets),'k',10);


%% run neural network
figure; hold on;
for i = 1:cv.NumTestSets

fprintf(1,'Training %dth fold ...\n',i);
    
% split data
inputs_training = selectMatrix(inputs,cv.training(i));
inputs_test = selectMatrix(inputs,cv.test(i));
targets_training = selectMatrix(targets,cv.training(i));
targets_test = selectMatrix(targets,cv.test(i));

% progressively use more data for training
N = 20;
size_block = cv.TrainSize(i)/N;
trainig_size = [];
training_err_rate = [];
test_err_rate = [];

%trainig_size = [1, 3, 6, 10:20:190, 200:size_block:cv.TrainSize(i)];
trainig_size = [1, 3, 6, 10:20:190, 200:50:500];

for j = 1:length(trainig_size)
   
 trainig_size(j) = min(floor(trainig_size(j)),cv.TrainSize(i));
   
% create network
hiddenLayerSize = 10;
clear net;
net = patternnet(hiddenLayerSize);

% remove constant inputs and map data to -1 to 1
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% data partition
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
%net.divideParam.valRatio = 10/100;
%net.divideParam.testRatio = 10/100;

% use scaled conjugate gradient backpropagation.
net.trainFcn = 'trainscg';
% cross-entropy as performance function
net.performFcn = 'crossentropy';  
% plot function
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};
   
% Train the Network
[net,tr] = train(net,inputs_training(:,1:trainig_size(j)), ...
                    targets_training(:,1:trainig_size(j)));
nntraintool('close');

% Test the Network
predictions_training = net(inputs_training);
predictions_test = net(inputs_test);

% calculate error rate
tind = vec2ind(targets_training);
pind = vec2ind(predictions_training);
training_err_rate(j) = sum(tind ~= pind)/numel(tind);
tind = vec2ind(targets_test);
pind = vec2ind(predictions_test);
test_err_rate(j) = sum(tind ~= pind)/numel(tind);

end

if i == 1 % initialize size
mean_training_err_rate = 0*training_err_rate;
mean_test_err_rate = 0*test_err_rate;
end

mean_training_err_rate = (mean_training_err_rate*(i-1)+training_err_rate)/i;
mean_test_err_rate = (mean_test_err_rate*(i-1)+test_err_rate)/i;
%plot(trainig_size, training_err_rate,'b-');
%plot(trainig_size, test_err_rate,'r-');
end

p(1) = plot(trainig_size, mean_training_err_rate,'b-o');
p(2) = plot(trainig_size, mean_test_err_rate,'r-o');

xlabel('Training Size',  'Fontsize', 16);
ylabel('Percentage Error',  'Fontsize', 16);
set(gcf, 'color', 'white');
set(gca, 'box', 'on', 'Fontsize', 14, 'linewidth', 1);
set(p,  'linewidth', 1.5, 'markersize', 8);
le = legend('Training Set', 'Test Set');
%set(le, 'box', 'off');
text(mean(trainig_size)/2, 0.3, 0, 'Neural Network - Mushroom Data','Fontsize', 16);
save;

save;

return;


% Recalculate Training, Validation and Test Performance
% performance = perform(net,targets,predictions);
% trainTargets = targets .* tr.trainMask{1};
% valTargets = targets  .* tr.valMask{1};
% testTargets = targets  .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,predictions);
% valPerformance = perform(net,valTargets,predictions);
% testPerformance = perform(net,testTargets,predictions);

% View the Network
%view(net)

% plot result
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,predictions)
% figure, plotroc(targets,predictions)


%% convert matrix array to normalized numeric matrix
function matrix = String2Matrix(stringMatrix)
    matrix = [];
    for i = 1:length(stringMatrix)  
      matrix = [matrix, String2Num(stringMatrix{i})];
    end
return


%% convert string array to normalized numeric array
function numArray = String2Num(stringArray)

    str = getStruct(stringArray);
    numArray =[];
    for i = 1:length(stringArray)
        if ~isempty(regexp(stringArray{i}(1),'[0-9]', 'once'))
           fieldname = ['number',stringArray{i}];
        else  
           fieldname = stringArray{i};
        end
        numArray(i,1) =  str.(fieldname);
    end
return


%% helper structure for converting string to numeric
 function str = getStruct(stringArray)
 str = struct();
 % assign numerical values
 n = 0;
 for i=1:length(stringArray)
     if ~isempty(regexp(stringArray{i}(1),'[0-9]', 'once'))
         fieldname = ['number',stringArray{i}];
     else
         fieldname = stringArray{i};
     end
     if ~isfield(str,fieldname)
         str.(fieldname)= n;
         n = n + 1;
     end
 end
 % normalize to [0,1]
 fields = fieldnames(str); 
 for i= 1:numel(fields)
     str.(fields{i}) = str.(fields{i})/max((n-1),1);
 end
return;


%% convert string array to binary classification matrix
function target = String2BinaryMaxtrix(stringArray)
    s = getStruct(stringArray);
    fields = fieldnames(s);
    for i = 1:numel(fields) 
       s.(fields{i}) = i;
    end
    index = [];
    for i=1:length(stringArray)
    index = [index, s.(stringArray{i})];
    end
    target = full(ind2vec(index,numel(fields)));
return
    

%% select matrix collumns based on id
function B = selectMatrix(A,id)
      B = [];
      for i = 1:length(id)
        if (id(i) ~= 0)
            B = [B, A(:,i)];
        end
      end
return



function newdata = shuffle(data)
      
    N = length(data); % # of collumn
    K = length(data{1}); % # of samples
    
    id = randperm(K); % random permutation
    newdata = {};
  for n = 1:N
      col = {};
    for i = 1:length(id)
       col = [col; data{n}(id(i))];
    end
     newdata = [newdata, {col}];
  end
  
return;

