import urllib


class PhotoInfo():
    """
    Class for photo information.
    """
    def __init__(self, photo_dict):
        """
        Initializer function
        :param photo_dict: The photo dictionary, which is a json dictionary.
        """
        self.photo_dict = photo_dict

    def get_id(self):
        """
        :return: Photo ID
        """
        return self.photo_dict.get('id', None)

    def get_title(self):
        """
        :return: Photo title
        """
        return self.photo_dict.get('title', None)

    def get_lat(self):
        """
        :return: Latitude where the photo was taken
        """
        return float(self.photo_dict.get('latitude', None))

    def get_long(self):
        """
        :return: Longitude where the photo was taken
        """
        return float(self.photo_dict.get('longitude', None))

    def get_coords(self):
        """
        :return: Coords where the photo was taken
        """
        return [self.get_lat(), self.get_long()]

    def get_accuracy(self):
        """
        :return: Accuracy of photo location
        """
        return int(self.photo_dict.get('accuracy', None))

    def get_tags(self):
        """
        :return: Returns the tags of the photo
        """
        tag_str = self.photo_dict.get('tags', None)
        if tag_str is None or len(tag_str) == 0:
            return []
        return tag_str.strip().split(' ')

    def get_url(self):
        """
        :return: url of photo
        """
        url = self.photo_dict.get('url_o')
        if url is None:
            url = f"https://farm{self.photo_dict.get('farm')}.staticflickr.com/{self.photo_dict.get('server')}" \
                  f"/{self.photo_dict.get('id')}_{self.photo_dict.get('secret')}_b.jpg"
        return url

    def download_to_file(self, filename):
        """
        Downloads the image to the specified filename path
        :param filename: the filename to be downloaded to
        :return: No return value
        """
        try:
            url = self.get_url()
            try:
                urllib.URLopener().retrieve(url, filename)
            except:
                urllib.request.urlretrieve(url, filename)
        except:
            pass

    def get_field(self, fieldname):
        """
        Get a custom field of the json dictionary
        :param fieldname: The name of the field
        :return: The value
        """
        return self.photo_dict.get(fieldname, None)
