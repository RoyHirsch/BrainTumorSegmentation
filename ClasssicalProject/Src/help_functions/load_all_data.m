function data = load_all_data()
str = which('dataBN.mat','-all');
data = struct();
for i=1:length(str)
    load(char(str(i)));
    data(i).f = double(im);
end
end

% loads all the 4D images to struct data
% 4D image may be called as: data(item).f where item is a numer between [1,30]
% image order:
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001/dataBN.mat
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0002/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0003/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0004/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0005/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0006/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0007/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0008/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0009/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0010/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0011/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0012/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0013/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0014/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0015/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0022/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0024/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0025/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0026/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0027/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0001/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0002/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0004/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0006/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0008/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0011/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0012/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0013/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0014/dataBN.mat  % Shadowed
% /Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_LG0015/dataBN.mat  % Shadowed
