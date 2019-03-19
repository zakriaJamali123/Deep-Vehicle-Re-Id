nQuery = 1678;
nTest = 11579;
cm = 'sudo caffe/Zak/extract_features ';

sn = 'apr_Model';


exfile = ' ex-mod-col-fc6.prototxt';

blobs = ' fc7,fc7-bn,fc-fc-part-1,fc-fc-part-2,';
blobs = [blobs 'fc-fc-part-3,concat-features-parts'];
blobs = [blobs ',att-branch-fc'];

datafi = ' features/fc7.data,features/fc7-bn.data,';
datafi = [datafi, 'features/p1.data,features/p2.data,'];
datafi = [datafi, 'features/p3.data,features/ps.data'];
datafi = [datafi, ',features/at.data'];

aha=system([cm, sn, exfile, blobs, datafi, ' 51 1']);
if aha>0
    fprintf('something wrong!!!!\n')
    return
end
verifeature = read_code('features/fc7.data', 13257, 4096);
verifeature_bn = read_code('features/fc7-bn.data',13257, 4096);
p1 = read_code('features/p1.data',13257,1024);
p2 = read_code('features/p2.data',13257,1024);
p3 = read_code('features/p3.data',13257,1024);

at = read_code('features/at.data',13257,512);

verifeature = norm_code(verifeature);
verifeature_bn = norm_code(verifeature_bn);
p1 = norm_code(p1);
p2 = norm_code(p2);
p3 = norm_code(p3);

at = norm_code(at);

verifeature = [verifeature; verifeature_bn;p1;p2;p3;at];
queryfeature = verifeature(:, 1:1678);
testfeature = verifeature(:, 1679:end);
dist = 1 - testfeature' * queryfeature;
maxgt = 256;
gt_index =  zeros(nQuery, maxgt);
fidin = fopen('gt_index.txt');

for i = 1:nQuery
    gt_index_line = fgetl(fidin);
    gt_line = str2num(gt_index_line);
    for j = 1:size(gt_line, 2)
       gt_index(i, j) = gt_line(j); 
    end
end

maxjk = 256;
jk_index = zeros(nQuery, maxjk);
fidin = fopen('jk_index.txt');
for i = 1:nQuery
    jk_index_line = fgetl(fidin);
    jk_line = str2num(jk_index_line);
    for j = 1:size(jk_line, 2)
       jk_index(i, j) = jk_line(j); 
    end
end
ap = zeros(nQuery, 1); 
CMC = zeros(nQuery, nTest);
r1 = 0;
parfor k = 1:nQuery
%     k
      good_index = reshape(gt_index(k,:), 1, []);
      good_index = good_index(good_index ~= 0);
      junk_index = reshape(jk_index(k,:), 1, []);
      junk_index = junk_index(junk_index ~= 0);
    score = dist(:, k);
    
    [~, index] = sort(score, 'ascend');  
    [ap(k), CMC(k, :)] = 
        compute_AP(good_index, junk_index, index);

end
CMC = mean(CMC);
fprintf(' mAP = %f,\nr1 = %f,\nr5 = %f\r\n',...
    mean(ap), CMC(1), CMC(5));

s = 50;
CMC_curve = CMC;

