var VG = (function(vg, $) {
    VG.getObj = function(ind, key, cb) {
        var that = (this === vg ? {} : this);
        var buck = 'visualaipascalparts';
        var s3Client = new AWS.S3();
        var params = {
            Bucket: buck,
            Key: key
        }
        s3Client.getObject(params, function(data, err) {
            cb(ind, data, err)
        });
    }


    return vg;
}(VG || {}, jQuery));