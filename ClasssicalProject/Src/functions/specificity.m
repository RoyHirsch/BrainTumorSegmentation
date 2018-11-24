function out = specificity(labels,predict)
out =  sum(sum(sum(and(not(labels),not(predict))))) / sum(sum(sum((not(labels)))));
end
