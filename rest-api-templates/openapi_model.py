#!/usr/bin/env python
# Asterisk -- An open source telephony toolkit.
#
# Copyright (C) 2013, Digium, Inc.
# Copyright (C) 2024, Asterisk Project
#
# David M. Lee, II <dlee@digium.com>
#
# See http://www.asterisk.org for more information about
# the Asterisk project. Please do not directly contact
# any of the maintainers of this project for assistance;
# the project provides a web site, mailing lists and IRC
# channels for your use.
#
# This program is free software, distributed under the terms of
# the GNU General Public License Version 2. See the LICENSE file
# at the top of the source tree.
#

"""OpenAPI 3.1 model parser for Asterisk ARI.

This module provides adapter classes that parse OpenAPI 3.1 YAML files and
expose them with an interface compatible with the existing Mustache templates
(originally designed for Swagger 1.x).
"""

from __future__ import print_function
import json
import os
import re
import sys
from collections import OrderedDict

try:
    import yaml
except ImportError:
    print("PyYAML required. Please pip install pyyaml.", file=sys.stderr)
    sys.exit(1)


# OpenAPI 3.1 type mappings to internal types
OPENAPI_TYPE_MAP = {
    'string': 'string',
    'integer': 'int',
    'number': 'double',
    'boolean': 'boolean',
    'object': 'object',
    'array': 'List',
}

# Format-specific type overrides
OPENAPI_FORMAT_MAP = {
    ('integer', 'int32'): 'int',
    ('integer', 'int64'): 'long',
    ('number', 'float'): 'float',
    ('number', 'double'): 'double',
    ('string', 'date'): 'Date',
    ('string', 'date-time'): 'Date',
    ('string', 'binary'): 'binary',
    ('string', 'byte'): 'byte',
}

# Primitives that don't need model references
PRIMITIVES = [
    'void', 'string', 'boolean', 'number', 'int', 'long',
    'double', 'float', 'Date', 'object', 'byte', 'binary',
]


class Stringify:
    """Simple mix-in to make the repr of the model classes more meaningful."""
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.__dict__)


class SwaggerType(Stringify):
    """Model of a data type - compatible with swagger_model.SwaggerType."""

    def __init__(self, name):
        self.name = name
        self.is_list = name.startswith('List[')
        if self.is_list:
            # Extract inner type from List[X]
            self.singular_name = name[5:-1]
        else:
            self.singular_name = name
        self.lc_singular_name = self.singular_name.lower()
        self.is_primitive = self.singular_name in PRIMITIVES
        self.is_binary = self.singular_name == 'binary'

        # These get set by processor
        self.c_name = None
        self.c_singular_name = None
        self.wiki_name = None


class ErrorResponse(Stringify):
    """Model of an error response."""

    def __init__(self, code, reason):
        self.code = code
        self.reason = reason
        # Set by processor
        self.wiki_reason = None


class AllowableList(Stringify):
    """Model of allowable values."""

    def __init__(self, values):
        self.values = values

    def to_wiki(self):
        return "Allowed values: %s" % ", ".join(str(v) for v in self.values)


class Parameter(Stringify):
    """Model of an operation parameter."""

    def __init__(self, name, param_type, data_type, description='',
                 required=False, default_value=None, allowable_values=None,
                 allow_multiple=False):
        self.name = name
        self.param_type = param_type
        self.data_type = data_type
        self.description = description or ''
        self.required = required
        self.default_value = default_value
        self.allowable_values = allowable_values
        self.allow_multiple = allow_multiple

        # Set by processor
        self.c_name = None
        self.c_data_type = None
        self.c_convert = None
        self.json_convert = None
        self.c_space = None
        self.wiki_description = None
        self.wiki_allowable_values = None
        self.is_body_parameter = param_type == 'body'

    def is_type(self, other_type):
        return self.param_type == other_type


class Property(Stringify):
    """Model of a schema property."""

    def __init__(self, name, type_obj, description='', required=False):
        self.name = name
        self.type = type_obj
        self.description = description or ''
        self.required = required

        # Set by processor
        self.wiki_description = None


class Model(Stringify):
    """Model of a schema/model definition."""

    def __init__(self, model_id, description=''):
        self.id = model_id
        self.id_lc = model_id.lower()
        self.description = description or ''
        self.subtypes = []
        self._subtype_types = []
        self._properties = []
        self._extends_type = None
        self._discriminator = None
        self.model_json = ''

        # Set by processor
        self.c_id = None
        self.description_dox = None
        self.wiki_description = None

    def set_properties(self, props):
        self._properties = sorted(props, key=lambda p: p.name)

    def set_discriminator(self, prop):
        self._discriminator = prop

    def set_extends_type(self, model):
        self._extends_type = model

    def set_subtype_types(self, subtypes):
        self._subtype_types = subtypes

    def properties(self):
        """Returns properties including inherited ones."""
        base_props = []
        if self._extends_type:
            base_props = self._extends_type.properties()
        return base_props + self._properties

    def has_properties(self):
        return len(self.properties()) > 0

    def extends(self):
        return self._extends_type.id if self._extends_type else None

    def extends_lc(self):
        return self._extends_type.id_lc if self._extends_type else None

    def discriminator(self):
        """Returns the discriminator, digging through base types if needed."""
        if self._discriminator:
            return self._discriminator
        if self._extends_type:
            return self._extends_type.discriminator()
        return None

    def has_subtypes(self):
        return len(self.subtypes) > 0

    def all_subtypes(self):
        """Returns the full list of all subtypes, including sub-subtypes."""
        res = list(self._subtype_types)
        for subtype in self._subtype_types:
            res.extend(subtype.all_subtypes())
        return sorted(res, key=lambda m: m.id)


class Operation(Stringify):
    """Model of an HTTP operation."""

    def __init__(self, http_method, nickname, path, summary='', notes=''):
        self.http_method = http_method.upper()
        self.nickname = nickname
        self.nickname_lc = nickname.lower()
        self.path = path
        self.summary = summary or ''
        self.notes = notes or ''
        self.parameters = []
        self.error_responses = []
        self.response_class = None
        self.is_websocket = False
        self.websocket_protocol = None
        self.since = ''

        # Computed
        self.is_req = True

        # Set by processor
        self.c_nickname = None
        self.c_http_method = None
        self.wiki_summary = None
        self.wiki_notes = None

    def _categorize_parameters(self):
        """Categorize parameters by type."""
        self.query_parameters = [p for p in self.parameters if p.is_type('query')]
        self.path_parameters = [p for p in self.parameters if p.is_type('path')]
        self.header_parameters = [p for p in self.parameters if p.is_type('header')]
        body_params = [p for p in self.parameters if p.is_type('body')]
        self.body_parameter = body_params[0] if body_params else None

        self.has_query_parameters = bool(self.query_parameters)
        self.has_path_parameters = bool(self.path_parameters)
        self.has_header_parameters = bool(self.header_parameters)
        self.has_body_parameter = bool(self.body_parameter)
        self.has_parameters = (self.has_query_parameters or
                               self.has_path_parameters or
                               self.has_header_parameters)
        self.has_error_responses = bool(self.error_responses)

        # parse_body is true if there's a body param or query params
        self.parse_body = (self.body_parameter or self.has_query_parameters) and True
        self.is_binary_response = (self.response_class and
                                   self.response_class.is_binary)


class Api(Stringify):
    """Model of an API path with its operations."""

    def __init__(self, path):
        self.path = path
        self.operations = []
        self.description = ''
        self.full_name = None

        # Set by processor
        self.wiki_path = None

    @property
    def has_websocket(self):
        return any(op.is_websocket for op in self.operations)


class PathSegment(Stringify):
    """Tree representation of API paths for REST handler generation."""

    def __init__(self, name, parent):
        # Remove {markers} from path segment
        if name.startswith('{') and name.endswith('}'):
            self.name = name[1:-1]
            self.is_wildcard = True
        else:
            self.name = name
            self.is_wildcard = None

        self._children = OrderedDict()
        self.operations = []

        if not self.name:
            assert not parent
            self.full_name = ''
        elif not parent or not parent.name:
            self.full_name = name.strip('{}')
        else:
            self.full_name = "%s_%s" % (parent.full_name, self.name)

    def get_child(self, path):
        """Walk descendants to get path, creating if necessary."""
        segment_name = path[0].strip('{}')
        if segment_name != self.name:
            # Handle case where names don't match due to wildcards
            if path[0].startswith('{'):
                segment_name = path[0][1:-1]
        assert segment_name == self.name, f"{segment_name} != {self.name}"

        if len(path) == 1:
            return self

        child_key = path[1]
        child = self._children.get(child_key)
        if not child:
            child = PathSegment(path[1], self)
            self._children[child_key] = child
        return child.get_child(path[1:])

    def children(self):
        return list(self._children.values())

    def num_children(self):
        return len(self._children)


class ApiDeclaration(Stringify):
    """Model of an API resource declaration."""

    def __init__(self, name):
        self.name = name
        self.base_path = 'http://localhost:8088/ari'
        self.resource_path = f'/api-docs/{name}.{{format}}'
        self.apis = []
        self.models = []
        self.requires_modules = []
        self.author = 'David M. Lee, II <dlee@digium.com>'
        self.copyright = 'Copyright (C) 2012 - 2013, Digium, Inc.'
        self.description = ''
        self.since = ''

        # Set by ResourceApi
        self.c_name = None

    @property
    def has_websocket(self):
        return any(api.has_websocket for api in self.apis)


class ResourceApi(Stringify):
    """Model of a resource API (maps to a tag in OpenAPI)."""

    def __init__(self, name, description=''):
        self.name = name
        self.description = description or ''
        self.api_declaration = None
        self.path = f'/api-docs/{name}.json'
        self.file = None

        # Set by processor
        self.name_caps = None
        self.name_title = None
        self.c_name = None
        self.wiki_prefix = None
        self.root_path = None
        self.root_full_name = None


class OpenAPISpec(Stringify):
    """Root model for an OpenAPI 3.1 specification."""

    def __init__(self):
        self.swagger_version = '1.2'  # Compat with templates
        self.api_version = '12.0.0'
        self.base_path = 'http://localhost:8088/ari'
        self.apis = []  # List of ResourceApi

    @classmethod
    def load_file(cls, openapi_file, processor):
        """Load and parse an OpenAPI 3.1 YAML file.

        Args:
            openapi_file: Path to openapi.yaml
            processor: AsteriskProcessor instance for post-processing

        Returns:
            OpenAPISpec instance with all APIs loaded
        """
        spec = cls()

        with open(openapi_file, 'r') as f:
            data = yaml.safe_load(f)

        info = data.get('info', {})
        spec.api_version = info.get('version', '12.0.0')
        servers = data.get('servers', [])
        if servers:
            spec.base_path = servers[0].get('url', spec.base_path)

        # Get global copyright/author from info (used as defaults)
        global_copyright = info.get('x-copyright')
        global_author = info.get('x-author')

        # Extract tags (these become ResourceApis) - keep full tag for extensions
        tags = {t['name']: t for t in data.get('tags', [])}

        # Group paths by tag
        paths_by_tag = {}
        for path, path_item in data.get('paths', {}).items():
            for method, op_data in path_item.items():
                if method in ('get', 'post', 'put', 'delete', 'patch'):
                    op_tags = op_data.get('tags', ['default'])
                    for tag in op_tags:
                        if tag not in paths_by_tag:
                            paths_by_tag[tag] = {}
                        if path not in paths_by_tag[tag]:
                            paths_by_tag[tag][path] = {}
                        paths_by_tag[tag][path][method] = op_data

        # Parse schemas/models
        schemas = data.get('components', {}).get('schemas', {})
        all_models = spec._parse_schemas(schemas, processor)

        # Get full paths data for x-requires-modules at path level
        full_paths = data.get('paths', {})

        # Create ResourceApi for each tag (preserve order from spec)
        for tag_name in paths_by_tag.keys():
            tag_info = tags.get(tag_name, {})
            tag_desc = tag_info.get('description', '') if isinstance(tag_info, dict) else ''
            resource_api = ResourceApi(tag_name, tag_desc)
            resource_api.api_declaration = spec._create_api_declaration(
                tag_name, tag_info, paths_by_tag[tag_name], all_models, processor, full_paths,
                global_copyright, global_author
            )
            spec.apis.append(resource_api)
            processor.process_resource_api(resource_api, [openapi_file])

        return spec

    def _parse_schemas(self, schemas, processor):
        """Parse all schema definitions into Model objects."""
        models = {}

        # First pass: create all models
        for schema_name, schema_data in schemas.items():
            # Skip certain utility schemas
            if schema_name == 'VariableBag':
                continue

            model = Model(schema_name, schema_data.get('description', ''))

            # Store raw JSON for model_json field
            model.model_json = json.dumps(schema_data, indent=2, separators=(',', ': '))

            # Track source API for model assignment
            model.source_api = schema_data.get('x-source-api')

            # Parse properties (from top-level and from allOf items)
            props = []
            all_properties = dict(schema_data.get('properties', {}))
            required_props = set(schema_data.get('required', []))

            # Also collect properties from allOf inline schemas
            for item in schema_data.get('allOf', []):
                if not item.get('$ref'):  # Skip refs, only process inline objects
                    all_properties.update(item.get('properties', {}))
                    required_props.update(item.get('required', []))

            # Find discriminator property name from this schema or allOf items
            disc_prop_name = None
            for item in schema_data.get('allOf', []):
                if not item.get('$ref'):
                    disc = item.get('discriminator', {})
                    if disc.get('propertyName'):
                        disc_prop_name = disc['propertyName']
                        break

            for prop_name, prop_data in all_properties.items():
                # Skip discriminator const values - these are inherited from parent
                if 'const' in prop_data:
                    continue
                # Skip discriminator enum values (added for OpenAPI UI, not for C code)
                # Only skip if property name matches the discriminator property
                if prop_data.get('enum') and prop_name == disc_prop_name:
                    continue
                type_obj = self._parse_type(prop_data)
                # Process the type to set c_name, etc.
                processor.process_type(type_obj, [schema_name, prop_name])
                prop = Property(
                    prop_name,
                    type_obj,
                    prop_data.get('description', ''),
                    prop_name in required_props
                )
                processor.process_property(prop, [schema_name, prop_name])
                props.append(prop)

            model.set_properties(props)

            # Handle discriminator
            discriminator = schema_data.get('discriminator')
            if discriminator:
                disc_prop_name = discriminator.get('propertyName')
                if disc_prop_name:
                    disc_props = [p for p in props if p.name == disc_prop_name]
                    if disc_props:
                        model.set_discriminator(disc_props[0])

            processor.process_model(model, [schema_name])
            models[schema_name] = model

        # Second pass: link inheritance (allOf patterns)
        for schema_name, schema_data in schemas.items():
            if schema_name not in models:
                continue
            model = models[schema_name]

            # Check for allOf (inheritance)
            all_of = schema_data.get('allOf', [])
            for item in all_of:
                ref = item.get('$ref')
                if ref:
                    parent_name = ref.split('/')[-1]
                    if parent_name in models:
                        parent = models[parent_name]
                        model.set_extends_type(parent)
                        if model.id not in parent.subtypes:
                            parent.subtypes.append(model.id)

        # Third pass: resolve subtype references
        for model in models.values():
            subtype_models = []
            for subtype_name in model.subtypes:
                if subtype_name in models:
                    subtype_models.append(models[subtype_name])
            model.set_subtype_types(subtype_models)

        return models

    def _parse_type(self, schema):
        """Parse a schema into a SwaggerType."""
        if '$ref' in schema:
            # Reference to another schema
            ref_name = schema['$ref'].split('/')[-1]
            return SwaggerType(ref_name)

        schema_type = schema.get('type', 'object')
        schema_format = schema.get('format')

        if schema_type == 'array':
            # Array type - get items type
            items = schema.get('items', {})
            if '$ref' in items:
                item_type = items['$ref'].split('/')[-1]
            else:
                item_type = self._get_simple_type(items)
            return SwaggerType(f'List[{item_type}]')

        # Check format-specific mapping
        if (schema_type, schema_format) in OPENAPI_FORMAT_MAP:
            return SwaggerType(OPENAPI_FORMAT_MAP[(schema_type, schema_format)])

        # Default type mapping
        return SwaggerType(OPENAPI_TYPE_MAP.get(schema_type, 'object'))

    def _get_simple_type(self, schema):
        """Get simple type name from schema."""
        schema_type = schema.get('type', 'object')
        schema_format = schema.get('format')

        if (schema_type, schema_format) in OPENAPI_FORMAT_MAP:
            return OPENAPI_FORMAT_MAP[(schema_type, schema_format)]

        return OPENAPI_TYPE_MAP.get(schema_type, 'object')

    def _create_api_declaration(self, tag_name, tag_info, paths, all_models, processor, full_paths,
                                 global_copyright=None, global_author=None):
        """Create an ApiDeclaration for a tag."""
        decl = ApiDeclaration(tag_name)
        if isinstance(tag_info, dict):
            decl.description = tag_info.get('description', '')
            # Use tag-level copyright/author if present, else fall back to global
            decl.copyright = tag_info.get('x-copyright') or global_copyright or decl.copyright
            decl.author = tag_info.get('x-author') or global_author or decl.author
        else:
            decl.description = tag_info or ''
            # Use global values if available
            if global_copyright:
                decl.copyright = global_copyright
            if global_author:
                decl.author = global_author

        # Collect models used by this API and required modules
        used_models = set()
        requires_modules = None  # Will be set from first path that has it

        # Group operations by path (preserve order from spec)
        apis_by_path = OrderedDict()
        for path, methods in paths.items():
            # Convert OpenAPI path to Swagger-style
            swagger_path = path  # Already in {param} format
            api = Api(swagger_path)

            # Get x-requires-modules from path level (take first, they should all be same)
            if requires_modules is None:
                path_item = full_paths.get(path, {})
                path_requires = path_item.get('x-requires-modules', [])
                if path_requires:
                    requires_modules = path_requires

            for method, op_data in methods.items():
                operation = self._parse_operation(
                    method, swagger_path, op_data, all_models, used_models, processor
                )
                api.operations.append(operation)

            processor.process_api(api, [tag_name, path])
            apis_by_path[swagger_path] = api

        decl.apis = list(apis_by_path.values())
        decl.requires_modules = requires_modules or []

        # Add models: ONLY those defined in this API's source file (matching Swagger behavior)
        # Models used by operations but defined elsewhere are handled by their source API
        decl.models = sorted(
            [m for name, m in all_models.items()
             if getattr(m, 'source_api', None) == tag_name],
            key=lambda m: m.id
        )

        return decl

    def _parse_operation(self, method, path, op_data, all_models, used_models, processor):
        """Parse an operation from OpenAPI data."""
        operation_id = op_data.get('operationId', '')
        # Split operationId like "channels_originate" -> nickname "originate"
        if '_' in operation_id:
            nickname = operation_id.split('_', 1)[1]
        else:
            nickname = operation_id

        summary = op_data.get('summary', '')
        description = op_data.get('description', '')

        operation = Operation(method, nickname, path, summary, description)

        # Parse version info
        since = op_data.get('x-version-added', '')
        if since:
            operation.since = since

        # Check for WebSocket upgrade
        if op_data.get('x-upgrade') == 'websocket':
            operation.is_websocket = True
            operation.is_req = False
            operation.websocket_protocol = op_data.get('x-websocket-protocol')

        # Parse parameters
        params = []
        for param_data in op_data.get('parameters', []):
            param = self._parse_parameter(param_data, processor)
            params.append(param)

        # Parse request body as body parameter
        request_body = op_data.get('requestBody')
        if request_body:
            body_param = self._parse_request_body(request_body, processor)
            if body_param:
                # Insert after specified parameter, or at start if null
                insert_after = request_body.get('x-param-insert-after')
                if insert_after is None:
                    # Insert at beginning
                    params.insert(0, body_param)
                else:
                    # Find the parameter to insert after
                    insert_idx = len(params)  # Default to end
                    for i, p in enumerate(params):
                        if p.name == insert_after:
                            insert_idx = i + 1
                            break
                    params.insert(insert_idx, body_param)

        operation.parameters = params

        # Parse response type (before _categorize_parameters so is_binary_response is set correctly)
        responses = op_data.get('responses', {})
        success_response = responses.get('200') or responses.get('201')
        if success_response:
            content = success_response.get('content', {})
            json_content = content.get('application/json', {})
            schema = json_content.get('schema', {})
            if schema:
                response_type = self._parse_type(schema)
                processor.process_type(response_type, [path, method, 'response'])
                operation.response_class = response_type

                # Track used models
                if response_type.is_list:
                    if response_type.singular_name not in PRIMITIVES:
                        self._collect_model_deps(response_type.singular_name, all_models, used_models)
                elif not response_type.is_primitive:
                    self._collect_model_deps(response_type.name, all_models, used_models)
        elif responses.get('204'):
            # 204 No Content = void response
            response_type = SwaggerType('void')
            processor.process_type(response_type, [path, method, 'response'])
            operation.response_class = response_type

        # Categorize parameters (after response_class is set for is_binary_response)
        operation._categorize_parameters()

        # Parse error responses
        for code, resp_data in responses.items():
            if code.startswith('4') or code.startswith('5'):
                reason = resp_data.get('description', '')
                error = ErrorResponse(int(code), reason)
                operation.error_responses.append(error)

        processor.process_operation(operation, [path, method])
        return operation

    def _parse_parameter(self, param_data, processor):
        """Parse a parameter from OpenAPI data."""
        name = param_data.get('name', '')
        param_in = param_data.get('in', 'query')
        description = param_data.get('description', '')
        required = param_data.get('required', False)

        # Get type from schema
        schema = param_data.get('schema', {})
        data_type = self._get_simple_type(schema)

        # Check for enum (allowable values)
        allowable_values = None
        enum = schema.get('enum')
        if enum:
            allowable_values = AllowableList(enum)

        # Check for array type (allow_multiple)
        allow_multiple = schema.get('type') == 'array'
        if allow_multiple:
            items = schema.get('items', {})
            data_type = self._get_simple_type(items)
            enum = items.get('enum')
            if enum:
                allowable_values = AllowableList(enum)

        # Get default value
        default_value = schema.get('default')

        param = Parameter(
            name=name,
            param_type=param_in,
            data_type=data_type,
            description=description,
            required=required,
            default_value=default_value,
            allowable_values=allowable_values,
            allow_multiple=allow_multiple
        )

        processor.process_parameter(param, [name])
        return param

    def _parse_request_body(self, request_body, processor):
        """Parse request body into a body parameter."""
        content = request_body.get('content', {})
        json_content = content.get('application/json', {})
        schema = json_content.get('schema', {})

        if not schema:
            return None

        description = request_body.get('description', '')
        # Use x-body-name extension if present, otherwise default to 'body'
        body_name = request_body.get('x-body-name', 'body')

        param = Parameter(
            name=body_name,
            param_type='body',
            data_type='object',
            description=description,
            required=request_body.get('required', False)
        )

        processor.process_parameter(param, [body_name])
        return param

    def _collect_model_deps(self, model_name, all_models, used_models):
        """Recursively collect model and its dependencies."""
        if model_name in used_models or model_name not in all_models:
            return

        used_models.add(model_name)
        model = all_models[model_name]

        # Add properties' types
        for prop in model.properties():
            if prop.type and not prop.type.is_primitive:
                type_name = prop.type.singular_name
                self._collect_model_deps(type_name, all_models, used_models)

        # Add parent type
        if model.extends():
            self._collect_model_deps(model.extends(), all_models, used_models)

        # Add subtypes
        for subtype in model.all_subtypes():
            self._collect_model_deps(subtype.id, all_models, used_models)

