//From https://www.exratione.com/2015/05/listing-large-s3-buckets-with-the-aws-sdk-for-node-js/

var VG= (function(vg, $) {
    //var AWS = require('aws-sdk');
    //var _ = require('underscore');
     
    // Create an S3 client.
    //
    // This will pick up the default credentials you have set up, such as
    // via a credentials file in the standard location, or environment
    // variables. See:
    // http://docs.aws.amazon.com/AWSJavaScriptSDK/guide/node-configuring.html
//    AWS.config.update({accessKeyId: 'AKIAIGY2E2BJRDZEKAHQ', secretAccessKey: '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', region: 'us-east-1'});
    var s3Client = new AWS.S3();
     
    // How many keys to retrieve with a single request to the S3 API.
    // Larger key sets require paging and multiple calls. 1000 is a 
    // sensible value for near all uses.
    var maxKeys = 1000;
     
    /**
     * List keys from the specified bucket.
     * 
     * If providing a prefix, only keys matching the prefix will be returned.
     *
     * If providing a delimiter, then a set of distinct path segments will be
     * returned from the keys to be listed. This is a way of listing "folders"
     * present given the keys that are there.
     *
     * @param {Object} options
     * @param {String} options.bucket - The bucket name.
     * @param {String} [options.prefix] - If set only return keys beginning with
     *   the prefix value.
     * @param {String} [options.delimiter] - If set return a list of distinct
     *   folders based on splitting keys by the delimiter.
     * @param {Function} callback - Callback of the form function (error, string[]).
     */
    vg.listKeys = function (options, callback) {
      var keys = [];
     
      /**
       * Recursively list keys.
       *
       * @param {String|undefined} marker - A value provided by the S3 API
       *   to enable paging of large lists of keys. The result set requested
       *   starts from the marker. If not provided, then the list starts
       *   from the first key.
       */
      function listKeysRecusively (marker) {
        options.marker = marker;
     
        listKeyPage(
          options,
          function (error, nextMarker, keyset) {
            if (error) {
              return callback(error, keys);
            }
     
            keys = keys.concat(keyset);
     
            if (nextMarker) {
              listKeysRecusively(nextMarker);
            } else {
              callback(null, keys);
            }
          }
        );
      }
     
      // Start the recursive listing at the beginning, with no marker.
      listKeysRecusively();
    }
     
    /**
     * List one page of a set of keys from the specified bucket.
     * 
     * If providing a prefix, only keys matching the prefix will be returned.
     *
     * If providing a delimiter, then a set of distinct path segments will be
     * returned from the keys to be listed. This is a way of listing "folders"
     * present given the keys that are there.
     *
     * If providing a marker, list a page of keys starting from the marker
     * position. Otherwise return the first page of keys.
     *
     * @param {Object} options
     * @param {String} options.bucket - The bucket name.
     * @param {String} [options.prefix] - If set only return keys beginning with
     *   the prefix value.
     * @param {String} [options.delimiter] - If set return a list of distinct
     *   folders based on splitting keys by the delimiter.
     * @param {String} [options.marker] - If set the list only a paged set of keys
     *   starting from the marker.
     * @param {Function} callback - Callback of the form 
        function (error, nextMarker, keys).
     */
    function listKeyPage (options, callback) {
      var params = {
        Bucket : options.bucket,
        Delimiter: options.delimiter,
        Marker : options.marker,
        MaxKeys : maxKeys,
        Prefix : options.prefix
      };
     
      s3Client.listObjects(params, function (error, response) {
        if (error) {
          return callback(error);
        } else if (response.err) {
          return callback(new Error(response.err));
        }
     
        // Convert the results into an array of key strings, or
        // common prefixes if we're using a delimiter.
        var img_names;
        if (options.delimiter) {
          // Note that if you set MaxKeys to 1 you can see some interesting
          // behavior in which the first response has no response.CommonPrefix
          // values, and so we have to skip over that and move on to the 
          // next page.
          //img_names = _.map(response.CommonPrefixes, function (item) {
//        //    return item.Prefix;
          //    return item.Key; // Editted by Will
          //});
          img_names = _.map(response.Contents, function (item) {
            return item.Key;
          });
        } else {
          img_names = _.map(response.Contents, function (item) {
            return item.Key;
          });
        }
     
        // Check to see if there are yet more keys to be obtained, and if so
        // return the marker for use in the next request.
        var nextMarker;
        if (response.IsTruncated) {
          if (options.delimiter) {
            // If specifying a delimiter, the response.NextMarker field exists.
            nextMarker = response.NextMarker;
          } else {
            // For normal listing, there is no response.NextMarker
            // and we must use the last key instead.
            nextMarker = img_names[img_names.length - 1];
          }
        }
     
        callback(null, nextMarker, img_names);
      });
    }
    return vg;
}(VG|| {}, jQuery));

