from django.apps import AppConfig

   


class QqConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'QQ'

    def ready(self):
        print("hi")