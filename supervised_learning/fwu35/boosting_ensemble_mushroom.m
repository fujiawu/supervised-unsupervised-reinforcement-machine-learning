%% clean the workspace and command line
function boosting_ensemble_mushroom

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

%% convert to normalized numeric and k-fold partition
Y = rawdata{1};
xdata = [rawdata(2:11), rawdata(13:23)];
cv = cvpartition(Y,'k',10);

%% run
figure; hold on;
mean_training_err_rate = 0;
mean_test_err_rate = 0;
for i = 1:cv.NumTestSets

fprintf(1,'Training %dth fold ...\n',i);
    
X_training = String2Matrix(xdata,cv.training(i));
Y_training = Y(cv.training(i));

X_test = String2Matrix(xdata,cv.test(i));
Y_test = Y(cv.test(i));

training_size = 1000;

% weak learner
t = templateTree('PruneCriterion','error','SplitCriterion','gdi','Prune','on');

% Ensemble learning with AdaBoostM1 boosting
 ens = fitensemble(X_training,Y_training, ...
                      'AdaBoostM1',training_size, t, 'LearnRate', 1, 'NPrint', 'off');

%plot(loss(ens,X_test,Y_test,'mode','cumulative'),'b-');
%mean(loss(ens,X_test,Y_test,'mode','cumulative'))

fprintf(1,'trained %d\n',ens.NumTrained);
training_err_rate = loss(ens,X_training,Y_training,'mode','cumulative');
test_err_rate = loss(ens,X_test,Y_test,'mode','cumulative');

mean_training_err_rate = (mean_training_err_rate*(i-1)+training_err_rate)/i;
mean_test_err_rate = (mean_test_err_rate*(i-1)+test_err_rate)/i;

end

p(1) = plot(mean_training_err_rate,'b-o');
p(2) = plot(mean_test_err_rate,'r-o');

xlabel('Training Size',  'Fontsize', 16);
ylabel('Percentage Error',  'Fontsize', 16);
set(gcf, 'color', 'white');
set(gca, 'box', 'on', 'Fontsize', 14, 'linewidth', 1);
set(p,  'linewidth', 1.5, 'markersize', 8);
le = legend('Training Set', 'Test Set');
%set(le, 'box', 'off');
text(200, 0.3, 0, 'Ensemble boosting - Mushroom Data','Fontsize', 16);
save;

return;


%% convert matrix array to normalized numeric matrix
function matrix = String2Matrix(stringMatrix,id)
    matrix = [];
    for i = 1:length(stringMatrix)  
      matrix = [matrix, String2Num(stringMatrix{i},id)];
    end
return


%% convert string array to normalized numeric array
function numArray = String2Num(stringArray,id)
    stringArray=stringArray(id);
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


