clear;
clc;
close all;


r = [0:0.01:1];
M = 1;

para = [];
for j = 1:M
    p.m = [rand, rand];
    p.sig = 0.1*[rand, rand];
    p.h = [rand];
    para = [para;p];
end


x1 = [];
x2 = [];
y = [];


for j = 1:length(r)
    for k = 1:length(r)
       x1(j,k) = r(j);
       x2(j,k) = r(k);
       y(j,k) = 0;
    end
end

for j = 1:length(r)
    for k = 1:length(r)
y(j,k) = y(j,k) + KHillsEvaluationFunction([x1(j,k), x2(j,k)], para);
    end
end

mesh(x1,x2,y)
