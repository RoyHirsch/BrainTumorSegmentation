function label = load_all_labels()
str = which('gt4.mat','-all');
label = struct();
for i=1:length(str)
    load(char(str(i)));
    label(i).f = double(gt4);
end
end