function ii=getscale(xv,x)

N=length(xv);

if N<=1
   ii=0;
   display('scale error');
   return;
end

if length(x)~=1
   ii=0;
   display('getscale input error');
   return;
end

if x<min(xv)
   ii=0;
   display('smaller than scale');
   return;
end
    
if x>max(xv)
   ii=0;
   x
   display('larger than scale');
   return;
end    
    
[~,ii] = min(abs(xv-x));

end