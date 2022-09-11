
n = 237;
dir_name = 'images';
out_file_name = 'test_lat_long.csv';
writelines('name,lat,long',out_file_name,WriteMode="append")
for i = 0:n-1
    file_name = strcat('image', int2str(i), '.jpg');
    path_name = strcat(dir_name, '/', file_name);
    inf = imfinfo(path_name).Comment;
    lat = 0;
    long = 0;
    for i = 1:length(inf)
        cell = inf(i);
        str = cell{1};
        len = strlength(str);
        if startsWith(str, 'latitude')
            num_str = extractAfter(str, 'latitude: ');
            lat = num_str;
        end
        if startsWith(str, 'longitude')
            num_str = extractAfter(str, 'longitude: ');
            long = num_str;
        end
    end
    writelines(strcat(file_name, ',', lat, ',', long),out_file_name,WriteMode="append")
end