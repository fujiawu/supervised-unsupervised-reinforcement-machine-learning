function ii = lookup(lib, item)

ii=0;

N = size(lib,1);
M = size(lib,2);

Ni = size(item,1);
Mi = size(item,2);

if M==0
    ii = 0;
    return;
end

if Ni~=1
    display('error:wrong item in lookup');
    ii = 0;
    return;
end

if Mi~=M
    display('error:wrong dimension in lookup');
    ii = 0;
    return;
end

for i=1:N
   flag = 1;
   for m=1:M
      if lib(i,m)~=item(m)
          flag = 0;
      end
   end
   if flag==1
       ii=i;
       return;
   end
   
end
