function out = dice(labels,predict)
out =  2*sum(sum(sum(and(labels,predict)))) / sum(sum(sum((labels + predict))));
end