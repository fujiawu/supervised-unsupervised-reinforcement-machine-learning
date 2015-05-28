
function y = KHillsEvaluationFunction(x, para)

y = 0;

N = length(x);
M = length(para);


for i = 1:M

h = para(i).h; %
m = para(i).m; % mean vector
sig = para(i).sig; % standard deviation vector


arg = 0;
for j = 1:N
   arg = arg + (x(j)-m(j))^2/(2*sig(j)^2);
end

y = y + h*exp(-arg);

end


end






