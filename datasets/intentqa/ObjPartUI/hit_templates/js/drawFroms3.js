var VG = (function(vg, $) {
    VG.myCtrl = function (key, $scope, $timeout) {    
        AWS.config.update({accessKeyId: 'AKIAIGY2E2BJRDZEKAHQ', secretAccessKey: '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', region: 'us-east-1'});

    var bucket = new AWS.S3({params: {Bucket: 'visualaipascalparts'}});

        bucket.getObject({Key:key},function(err,file){

        $timeout(function(){
            $scope.s3url = "data:image/jpeg;base64," + encode(file.Body);
        },1);
    });
    }

    function encode(data)
    {
        var str = data.reduce(function(a,b){ return a+String.fromCharCode(b) },'');
        return btoa(str).replace(/.{76}(?=.)/g,'$&\n');
    }

    return vg;
    
}(VG || {}, jQuery));

