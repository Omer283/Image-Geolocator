from flickrapi import FlickrAPI
import os

from photo_info import PhotoInfo
from PIL import Image
from torchvision import transforms
key = '' #Get your own key!
secret = ''

YEAR = 31536000
PRINT_EVERY = 500
PRINT_ERRORS = False
NEG_QUERIES = '-blackandwhite -monochrome -birthday -party -parties -portrait -bw -abstract -macro -me -wedding -indoors -fun -kid -child -children -graffiti -prom -concert -friend -family -dog -cat -face -cameraphone -woman -women -girl -lady -boy -guy -nude -fraternity -frat -gay -lesbian -live -baby -stilllife'


def get_photos(max_count, max_nodup=3800, per_page=200, image_tag=None, is_geotagged=True, image_name=None):
    '''
    Returns an array of PhotoInfo s matching the specified parameters
    :param max_count: The amount of PhotoInfo s
    :param max_nodup: The amount of images to scrape in a time interval
    :param per_page: Images to scrape per page
    :param image_tag: Tags to use in the search
    :param is_geotagged: Should the images be geotagged?
    :param image_name: Search by name
    :return: Array of PhotoInfo s
    '''
    global key, secret
    flickr = FlickrAPI(key, secret, format='parsed-json')
    count = 0
    geotag_bool = 0
    if is_geotagged:
        geotag_bool = 1
    DELTA = YEAR // 8 #TODO play
    min_upload = 1659822707 - DELTA - 5
    max_upload = 1659822707
    photos = []
    while count < max_count:
        min_upload -= DELTA
        max_upload -= DELTA
        print(f'So far, scraped {count} photos')
        for i in range(max_nodup // per_page):
            if count >= max_count:
                break
            try:
                photos_json = flickr.photos.search(api_key=key,
                                                sort='relevance',
                                                tag_mode='all',
                                                tags=image_tag,
                                                text=image_name,
                                                has_geo=geotag_bool,
                                                extras='description,geo,tags,url_o',
                                                per_page=per_page,
                                                page=i+1,
                                                min_upload_date=str(min_upload),
                                                max_upload_date=str(max_upload),
                                                accuracy=6,
                                                geo_context=2)
                json_list = photos_json['photos']['photo']
                for photo in json_list:
                    photos.append(PhotoInfo(photo))
                    count += 1
                    if count >= max_count:
                        break
            except:
                if PRINT_ERRORS:
                    print(f'Scrape error at count {count}')
    print('Done scraping')
    return photos


def download_all(photos, dir_name='geo_images', print_every=PRINT_EVERY, offset=0):
    '''
    Downloads an array of PhotoInfo s to a specific directory
    :param photos: PhotoInfo s to be downloaded
    :param dir_name: Target directory
    :param print_every: Prints progress every x downloads
    :param offset: If you want to index the images from a certain offset
    :return: List of PhotoInfos that were actually downloaded
    (images can be discarded due to being huge or grayscale)
    '''
    dir = os.getcwd() + os.sep + dir_name + os.sep  # save directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    url_counter = 0
    photo_list = []
    tr = transforms.ToTensor()
    for photo in photos:
        filename = dir + '/image' + str(url_counter + offset) + '.jpg'
        photo.download_to_file(filename)
        try:
            im = tr(Image.open(filename))
            if (im.shape)[0] != 3:
                print('Skipped grayscale')
                continue
        except:
            print('Huge image, skipping')
            continue
        photo_list.append(photo)
        url_counter += 1
        if url_counter % print_every == 0:
            print(f'Downloaded {url_counter}/{len(photos)} images')
    print('Done downloading')
    return photo_list


def scrape(max_count, csv_file_name, name=None, tag=NEG_QUERIES,
           is_geolocated=True, print_every=500,dir_name='geo_images',offset=0):
    '''
    Main function for scraping images. Scrapes and downloads images to filename and directory
    :param max_count: The amount of images to be scraped
    :param csv_file_name: Target file
    :param name: Name of images to be scraped
    :param tag: Tags of images to be scraped
    :param is_geolocated: Boolean flag, scrape geolocated images
    :param print_every: Print update every x downloads
    :param dir_name: Name of target directory
    :param offset: Offset to be downloaded from
    :return: None
    '''
    global PRINT_EVERY
    PRINT_EVERY = print_every
    photos = get_photos(max_count,image_name=name,image_tag=tag, is_geotagged=is_geolocated)
    checkpoint_size = 2000 #For safety
    cur_uploaded = 0
    idx = 0
    f = open(csv_file_name, 'w+')
    f.write('name,lat,long\n')
    while cur_uploaded < len(photos):
        until = min(cur_uploaded + checkpoint_size, len(photos))
        cur_photos = photos[cur_uploaded:until]
        updated_photos = download_all(cur_photos, dir_name=dir_name, offset=cur_uploaded+offset)
        for photo in updated_photos:
            name = 'image' + str(idx + offset) + '.jpg'
            data_list = []
            data_list.append(name)
            data_list.append(str(photo.get_lat()))
            data_list.append(str(photo.get_long()))
            # data_list.append(photo.get_url()) #optional
            f.write(','.join(data_list))
            f.write('\n')
            idx += 1
        cur_uploaded = until
        print(f'Downloaded {cur_uploaded} photos')