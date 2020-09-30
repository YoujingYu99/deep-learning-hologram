clc;
clear all;

rootdir='C:\Users\Youjing Yu\PycharmProjects\deep-learning-cgh\coco\original\gray';
subdir=dir(rootdir);
for i=1:length(subdir)
    subdirpath=fullfile(rootdir,subdir(i).name,'*.jpg');
    images=dir(subdirpath);
    for j=1:length(images)
        ImageName=fullfile(rootdir,subdir(i).name,images(j).name);
        a=imread(ImageName);

        %fresnel diffraction;
        a=im2double(a);
        [n,m]=size(a);
        lam=632e-9;
        pitch=360e-7;
        k=2*pi/lam;
        lx=m*pitch;
        ly=n*pitch;
        [x,y]=meshgrid(linspace(-lx/2,lx/2,m),linspace(-ly/2,ly/2,n));
        z=0.13;
        len=exp((1i*k)/(2*z)*(x.^2+y.^2));
        al=a.*len;
        resample=lam*z/pitch;
        [u,v]=meshgrid(linspace(-resample/2,resample/2,m),linspace(-resample/2,resample/2,n));
        b=fftshift(fft2(fftshift(al))).*exp(1i*k/(2*z)*(u.^2+v.^2));

        saveddir='C:\Users\Youjing Yu\PycharmProjects\deep-learning-cgh\coco\original\diffracted4';
        savedname=fullfile(saveddir,['diff',images(j).name]);
        b=uint8(abs(b));
        imwrite(abs(b),savedname,'jpg');  
    end
end



