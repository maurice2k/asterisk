#
# Asterisk -- An open source telephony toolkit.
#
# Copyright (C) 2013, Digium, Inc.
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

"""Processor which adds fields needed to generate Asterisk RESTful HTTP
binding code.

Works with both the old swagger_model.py and new openapi_model.py parsers.
"""

import os
import re

try:
    from collections import OrderedDict
except ImportError:
    from odict import OrderedDict


class ProcessorError(Exception):
    """Raised when an error is encountered during processing."""

    def __init__(self, msg, context=None):
        super().__init__(msg)
        self.context = context


def simple_name(name):
    """Removes the {markers} from a path segment.

    @param name: Path segment, with {pathVar} markers.
    """
    if name.startswith('{') and name.endswith('}'):
        return name[1:-1]
    return name


def wikify(text):
    """Escapes a string for the wiki.

    @param text: String to escape
    """
    if not text:
        return ''
    # Replace all line breaks with line feeds
    text = re.sub(r'<br\s*/?>', '\n', text)
    return re.sub(r'([{}\[\]])', r'\\\1', text)


def snakify(name):
    """Helper to take a camelCase or dash-separated name and make it
    snake_case.
    """
    r = ''
    prior_lower = False
    for c in name:
        if c.isupper() and prior_lower:
            r += "_"
        if c == '-':
            c = '_'
        prior_lower = c.islower()
        r += c.lower()
    return r


class PathSegment:
    """Tree representation of an API declaration for REST handler generation."""

    def __init__(self, name, parent):
        """Ctor.

        @param name: Name of this path segment. May have {pathVar} markers.
        @param parent: Parent PathSegment.
        """
        #: Segment name, with {pathVar} markers removed
        self.name = simple_name(name)
        #: True if segment is a {pathVar}, else None.
        self.is_wildcard = None
        #: Underscore separated name all ancestor segments
        self.full_name = None
        #: Dictionary of child PathSegments
        self._children = OrderedDict()
        #: List of operations on this segment
        self.operations = []

        if self.name != name:
            self.is_wildcard = True

        if not self.name:
            assert not parent
            self.full_name = ''
        if not parent or not parent.name:
            self.full_name = name
        else:
            self.full_name = "%s_%s" % (parent.full_name, self.name)

    def get_child(self, path):
        """Walks descendants to get path, creating it if necessary.

        @param path: List of path names.
        @return: PathSegment corresponding to path.
        """
        assert simple_name(path[0]) == self.name
        if len(path) == 1:
            return self
        child = self._children.get(path[1])
        if not child:
            child = PathSegment(path[1], self)
            self._children[path[1]] = child
        return child.get_child(path[1:])

    def children(self):
        """Gets list of children."""
        return list(self._children.values())

    def num_children(self):
        """Gets count of children."""
        return len(self._children)

    def __repr__(self):
        return "PathSegment(name=%r, full_name=%r)" % (self.name, self.full_name)


class AsteriskProcessor:
    """Processor which adds fields needed to generate Asterisk RESTful HTTP
    binding code.
    """

    #: How types map to C.
    type_mapping = {
        'string': 'const char *',
        'boolean': 'int',
        'number': 'int',
        'int': 'int',
        'long': 'long',
        'double': 'double',
        'float': 'float',
    }

    #: String conversion functions for string to C type.
    convert_mapping = {
        'string': '',
        'int': 'atoi',
        'long': 'atol',
        'double': 'atof',
        'boolean': 'ast_true',
    }

    #: JSON conversion functions
    json_convert_mapping = {
        'string': 'ast_json_string_get',
        'int': 'ast_json_integer_get',
        'long': 'ast_json_integer_get',
        'double': 'ast_json_real_get',
        'boolean': 'ast_json_is_true',
    }

    def __init__(self, wiki_prefix=''):
        self.wiki_prefix = wiki_prefix

    def process_resource_api(self, resource_api, context):
        """Process a ResourceApi, adding computed fields."""
        resource_api.wiki_prefix = self.wiki_prefix

        # Derive resource name - either from path or use existing name
        if hasattr(resource_api, 'path') and resource_api.path:
            resource_api.name = re.sub(r'\..*', '',
                                       os.path.basename(resource_api.path))
        # Now in all caps, for include guard
        resource_api.name_caps = resource_api.name.upper()
        resource_api.name_title = resource_api.name.capitalize()
        resource_api.c_name = snakify(resource_api.name)

        # Also set c_name on api_declaration for template access
        if resource_api.api_declaration:
            resource_api.api_declaration.c_name = resource_api.c_name

        # Construct the PathSegment tree for the API.
        if resource_api.api_declaration:
            resource_api.root_path = PathSegment('', None)
            for api in resource_api.api_declaration.apis:
                segment = resource_api.root_path.get_child(api.path.split('/'))
                for operation in api.operations:
                    segment.operations.append(operation)
                api.full_name = segment.full_name

            # Since every API path should start with /[resource], root should
            # have exactly one child.
            if resource_api.root_path.num_children() != 1:
                raise ProcessorError(
                    "Should not mix resources in one API declaration", context)
            # root_path isn't needed any more
            resource_api.root_path = list(resource_api.root_path.children())[0]
            if resource_api.name != resource_api.root_path.name:
                raise ProcessorError(
                    "API declaration name should match: %s != %s" % (
                        resource_api.name, resource_api.root_path.name), context)
            resource_api.root_full_name = resource_api.root_path.full_name

    def process_api(self, api, context):
        """Process an Api, adding computed fields."""
        api.wiki_path = wikify(api.path)

    def process_operation(self, operation, context):
        """Process an Operation, adding computed fields."""
        # Nicknames are camelCase, Asterisk coding is snake case
        operation.c_nickname = snakify(operation.nickname)
        operation.c_http_method = 'AST_HTTP_' + operation.http_method.upper()

        # Validate summary ends with period
        if operation.summary and not operation.summary.endswith("."):
            raise ProcessorError("Summary should end with .: %s" % operation.summary, context)

        operation.wiki_summary = wikify(operation.summary or "")
        operation.wiki_notes = wikify(operation.notes or "")

        # Process error responses
        for error_response in operation.error_responses:
            error_response.wiki_reason = wikify(error_response.reason or "")

        # Ensure parse_body is set
        if not hasattr(operation, 'parse_body'):
            operation.parse_body = (
                (hasattr(operation, 'body_parameter') and operation.body_parameter) or
                (hasattr(operation, 'has_query_parameters') and operation.has_query_parameters)
            ) and True

    def process_parameter(self, parameter, context):
        """Process a Parameter, adding computed fields."""
        if parameter.param_type == 'body':
            parameter.is_body_parameter = True
            parameter.c_data_type = 'struct ast_json *'
            parameter.c_convert = ''
            parameter.json_convert = ''
        else:
            parameter.is_body_parameter = False
            if parameter.data_type not in self.type_mapping:
                raise ProcessorError(
                    "Invalid parameter type %s" % parameter.data_type, context)
            # Type conversions
            parameter.c_data_type = self.type_mapping[parameter.data_type]
            parameter.c_convert = self.convert_mapping.get(parameter.data_type, '')
            parameter.json_convert = self.json_convert_mapping.get(parameter.data_type, '')

        # Parameter names are camelCase, Asterisk convention is snake case
        parameter.c_name = snakify(parameter.name)

        # You shouldn't put a space between 'char *' and the variable
        if parameter.c_data_type.endswith('*'):
            parameter.c_space = ''
        else:
            parameter.c_space = ' '

        parameter.wiki_description = wikify(parameter.description)

        if parameter.allowable_values:
            parameter.wiki_allowable_values = parameter.allowable_values.to_wiki()
        else:
            parameter.wiki_allowable_values = None

    def process_model(self, model, context):
        """Process a Model, adding computed fields."""
        model.description_dox = model.description.replace('\n', '\n * ')
        model.description_dox = re.sub(r' *\n', '\n', model.description_dox)
        model.wiki_description = wikify(model.description)
        model.c_id = snakify(model.id)
        return model

    def process_property(self, prop, context):
        """Process a Property, adding computed fields."""
        if "-" in prop.name:
            raise ProcessorError("Property names cannot have dashes", context)
        if prop.name != prop.name.lower():
            raise ProcessorError("Property name should be all lowercase: %s" % prop.name,
                                 context)
        prop.wiki_description = wikify(prop.description)

    def process_type(self, swagger_type, context):
        """Process a SwaggerType, adding computed fields."""
        swagger_type.c_name = snakify(swagger_type.name)
        swagger_type.c_singular_name = snakify(swagger_type.singular_name)
        swagger_type.wiki_name = wikify(swagger_type.name)

    def process_resource_listing(self, resource_listing, context):
        """Process the overall ResourceListing/OpenAPISpec object."""
        pass  # No additional processing needed at top level
