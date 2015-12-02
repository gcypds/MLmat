function [W,b,linInd] = netInitDownsampler(siz,sc,L)

numModules = zeros(1,L);
for l = 1:L;
  y = cell(1,numel(siz));
  numModules(l) = 1;    
  for d = 1:numel(siz)
    y{d} = round(sc/2):sc:siz(d);    
    numModules(l) = numModules(l)*numel(y{d});    
  end
  if numModules(l) == 1
    break;
  end
  [x{1},x{2}] = ndgrid(1:siz(1),1:siz(2));
  [y{1},y{2}] = ndgrid(y{1},y{2});
  ind{l} = sub2ind(siz,y{1},y{2});
  rows = zeros(1,numel(x{1}));
  for r=1:numel(y{1})
    D = pdist2([y{1}(r) y{2}(r)],[x{1}(:) x{2}(:)],'cityblock');
    rows(D<sc) = r;
  end
  tmp = reshape(rands(prod(siz)),siz);  
  tmp = tmp(rows>0);    
  %nw begin
  tmp_norm = sum(tmp.^2);
  beta = 0.7*numel(y{1}).^(1/numel(x{1}));
  tmp = beta*tmp/tmp_norm;
  %nw end
  cols = 1:numel(x{1});  
  cols = cols(rows>0);
  rows = rows(rows>0);  
  W{l} = sparse(rows(:),cols(:),tmp,numel(y{1}),numel(x{1}));          
  b{l} = beta*rands(numel(y{1}))/tmp_norm;  
  linInd{l} = sub2ind([numel(y{1}),numel(x{1})],rows(:),cols(:));
  siz = size(ind{l});
end