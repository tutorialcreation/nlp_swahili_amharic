from rest_framework import serializers
from . import models


class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Audio
        fields = ('id','audio_file',) 