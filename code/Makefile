OCCLUM_INFERENCE_IMGS := \
	cmli-occlum-inference-mnist \
	cmli-occlum-inference-resnet

INFERENCE_IMGS := \
	cmli-local-inference \
	$(OCCLUM_INFERENCE_IMGS)

.PHONY += all
all: $(INFERENCE_IMGS)

cmli-cprogs: cmli-ort
$(INFERENCE_IMGS): cmli-cprogs cmli-data cmli-ort
$(OCCLUM_INFERENCE_IMGS): cmli-occlum

# 1: image name
# 2: dockerfile
# 3: extra build args
define IMG_TEMPLATE =
.PHONY += cmli-$(1)
cmli-$(1): docker-logs
	sudo docker build $(3) -t cmli-$(1) -f $(2) ctx >docker-logs/$(1) 2>&1
endef

define SIMPLE_TEMPLATE =
$$(eval $$(call IMG_TEMPLATE,$(1),Dockerfile.$(1),))
endef
$(foreach i,cprogs data local-inference occlum ort,$(eval $(call SIMPLE_TEMPLATE,$(i))))

define OCCLUM_INFERENCE_TEMPLATE =
$$(eval $$(call IMG_TEMPLATE,occlum-inference-$(1),Dockerfile.occlum-inference,--build-arg=DATASET_NAME=$(1)))
endef
$(foreach d,mnist resnet,$(eval $(call OCCLUM_INFERENCE_TEMPLATE,$(d))))

docker-logs:
	mkdir "$@"
