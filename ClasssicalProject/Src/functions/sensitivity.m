function out = sensitivity(labels,predict)
out =  sum(sum(sum(and(labels,predict)))) / sum(sum(sum((labels))));
end
