var VG = (function(vg, $) {
    VG.putObj = function() {
        console.log('Submitting in VG.putObj');
        var file = document.getElementById('output').value; 
        var resultsTime = document.getElementById('results-time').value;
        var uploadFileName = 'points_task_'+resultsTime + '.json'
        var buck = 'visualaipascalresponses';
        var s3Client = new AWS.S3();
        var params = {
            Bucket: buck,
            Key: uploadFileName,
            Body: file,
        }
        s3Client.putObject(params, function(err, data) {
           if (err) console.log(err, err.stack); // an error occurred
           else     console.log(data);           // successful response   
        });
    }

    return vg;
}(VG|| {}, jQuery));

