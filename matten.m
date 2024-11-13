function ten = matten(D,dim,mode_n)
% put the index of n-mode dim to the first place
length_dim = length(dim);
sequ_rsp = 1:length_dim;
sequ_rsp(mode_n) = [];
sequ_rsp = [mode_n,sequ_rsp];
% get the sequence of permuting the dim sequence: 1,2,...,n
[~,sequ_pmt,~] = unique(sequ_rsp);
% reshape the tensor, n-mode first
ten = reshape(D,dim(sequ_rsp));
% permute the tensor to the sorted sequence
ten = permute(ten,sequ_pmt);
end